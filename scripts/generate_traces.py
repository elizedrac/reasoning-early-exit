"""
Generate reasoning traces with DeepSeek-R1-Distill-Qwen-1.5B.

Backends:
    --backend vllm   NVIDIA GPU (e.g. Colab T4), batched
    --backend mlx    Apple silicon, one question at a time, same Hugging Face weights
                     (~29 tokens/s on an M4: fine for tests, too slow for the full run)
Both use the pilot's prompt and chat template, temperature 0.6, top_p 0.95, default
--max-tokens 16384, and capture the generated token IDs directly.

Truncated traces (hit --max-tokens) get a greedy forced answer: append
"\\n</think>\\n\\nFinal answer: \\\\boxed{" to prompt + trace, decode up to 32 tokens, and take the
boxed answer. Chunking and labeling happen afterwards in scripts/label_chunks.py.

Resumable: results are appended as they finish and IDs already in --out are skipped.

Usage:
    python scripts/generate_traces.py --input data/trace_inputs/gsm8k_train.jsonl --out traces/gsm8k_train.jsonl
    python scripts/generate_traces.py --backend mlx --input ... --out ... --max-tokens 2048

Output: one JSON line per question with id, dataset, split, problem, gold, prompt_token_ids,
token_ids, num_tokens, finish_reason, truncated, trace, final_answer, final_correct,
forced_answer, forced_correct.
"""
import argparse
import json
import os
import re
import time
import zlib
from bisect import bisect_right

from math_verify import parse, verify

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
PROMPT = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
FORCE_SUFFIX = "\n</think>\n\nFinal answer: \\boxed{"
FORCE_MAX_TOKENS = 32

# ---------------------------------------------------------------------------
# Answer extraction and judging (from the pilot; the chunk-level regex labels are replaced by
# label_chunks.py, but the final answer after </think> still uses extract_answer)
# ---------------------------------------------------------------------------

# Reflection markers where a new chunk starts
MARKERS = re.compile(
    r"(?=\b(?:Wait|But wait|Alternatively|Hmm|Let me double-check|Let me verify|Let me check)\b)")

# Patterns for finding a stated answer at the end of a chunk
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?"
REJECTED = re.compile(
    r"\b(?:wrong|incorrect|mistake|can't be|cannot be|doesn't work|does not work|contradiction)\b", re.I)
CONCLUSIONS = [
    re.compile(r"\banswer\s*(?:is|should be|would be|must be|:)\s*(.+)$", re.I),
    re.compile(r"(\S+)\s+is the (?:final |correct )?answer\b", re.I),
    re.compile(r"\b(?:so|thus|therefore|hence)\b.*\b(?:is|are|equals|gives|totaling|totals)\s+(.+)$", re.I),
    re.compile(r"(?<![<>!])=\s*([^=]+)$"),
]


def last_boxed(text):
    start = text.rfind("\\boxed")
    if start == -1:
        return None
    i = text.find("{", start)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j].strip()
    return None


def clean_answer(text):
    """Tidy a candidate answer. Return None if it reads like prose or is too long."""
    ans = re.sub(r"\\[()\[\]]|\*\*|\\\$|\$", "", text).rsplit("=", 1)[-1].strip(" .,;:")
    number = re.match(rf"({NUMBER})(?:\s+[A-Za-z]+){{0,3}}$", ans)
    if number:
        return number.group(1).replace(",", "")
    ans = re.sub(r"sqrt\(([^()]+)\)", r"\\sqrt{\1}", ans)
    if not ans or len(ans) > 40 or re.search(r"[A-Za-z]{2,}", re.sub(r"\\[A-Za-z]+", "", ans)):
        return None
    return ans


def extract_answer(text):
    """Answer a chunk clearly states: last \\boxed{}, or a conclusion in its last two sentences.

    Returns None when the chunk states no clear answer, so it stays unlabeled.
    """
    boxed = last_boxed(text)
    if boxed:
        return boxed
    tail = [s.strip() for s in SENTENCE_BREAK.split(text) if s.strip()][-2:]
    if any(REJECTED.search(s) for s in tail):
        return None
    for sentence in reversed(tail):
        if sentence.endswith("?") or ", and " in sentence:
            continue
        for pattern in CONCLUSIONS:
            match = pattern.search(sentence)
            answer = clean_answer(match.group(1)) if match else None
            if answer:
                return answer
    return None


def is_correct(pred, gold):
    """True or False if both answers parse; None if they can't be compared."""
    try:
        parsed_gold = parse(f"\\boxed{{{gold}}}")
        parsed_pred = parse(f"\\boxed{{{pred}}}")
        if not parsed_gold or not parsed_pred:
            return None
        return bool(verify(parsed_gold, parsed_pred))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Character offsets -> token indices
# ---------------------------------------------------------------------------

def decode(tok, ids):
    return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def bytes_to_unicode():
    """GPT-2 byte-level BPE table (byte -> printable char), as used by Qwen tokenizers."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def token_byte_ends(tok, ids):
    """ends[t] = number of UTF-8 bytes in the decoded text of ids[:t] (len(ends) == len(ids) + 1).

    Works for byte-level BPE tokenizers (Qwen): each token string maps back to raw bytes.
    Special tokens (e.g. EOS) decode to nothing, matching skip_special_tokens=True.
    """
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    special = set(tok.all_special_ids)
    ends, total = [0], 0
    for i, piece in zip(ids, tok.convert_ids_to_tokens(ids)):
        if i in special:
            n = 0
        elif all(ch in byte_decoder for ch in piece):
            n = len(piece)  # one byte per byte-level character
        else:
            n = len(piece.encode("utf-8"))  # added tokens stored as plain text
        total += n
        ends.append(total)
    return ends


def char_to_token(trace, ends, char_pos):
    """Largest t such that tokens [0, t) decode to at most trace[:char_pos] (rounds down)."""
    byte_pos = len(trace[:char_pos].encode("utf-8"))
    return bisect_right(ends, byte_pos) - 1


def squash(s):
    return re.sub(r"\s+", "", s)


def add_token_spans(tok, trace, token_ids, chunks, rec_id, log):
    """Fill start_token/end_token for each chunk and check the decoded span against the text."""
    ends = token_byte_ends(tok, token_ids)
    full = decode(tok, token_ids)
    if full != trace:
        log({"id": rec_id, "k": None, "issue": "decode(token_ids) != trace text",
             "decoded_len": len(full), "trace_len": len(trace)})
    n_bad = 0
    for c in chunks:
        c["start_token"] = char_to_token(trace, ends, c["start_char"])
        c["end_token"] = char_to_token(trace, ends, c["end_char"])
        span = decode(tok, token_ids[c["start_token"]:c["end_token"]])
        try:
            assert squash(span) == squash(c["text"]), "token span does not match chunk text"
        except AssertionError as e:
            n_bad += 1
            log({"id": rec_id, "k": c["k"], "issue": str(e), "start_token": c["start_token"],
                 "end_token": c["end_token"], "chunk_head": c["text"][:80], "span_head": span[:80],
                 "chunk_tail": c["text"][-80:], "span_tail": span[-80:]})
    return n_bad


# ---------------------------------------------------------------------------
# Generation backends
# ---------------------------------------------------------------------------

def chat_prompt(tok, problem):
    return tok.apply_chat_template([{"role": "user", "content": PROMPT.format(problem=problem)}],
                                   tokenize=False, add_generation_prompt=True)


def force_suffix(trace):
    # If </think> was already emitted, don't add a second one.
    return FORCE_SUFFIX if "</think>" not in trace else FORCE_SUFFIX.replace("\n</think>", "")


class VLLMBackend:
    def __init__(self, args):
        from vllm import LLM, SamplingParams
        # Room for prompt + trace + forced-answer suffix and tokens.
        self.llm = LLM(model=MODEL, dtype=args.dtype, max_model_len=args.max_tokens + 2048, seed=args.seed)
        self.tok = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_tokens, seed=args.seed)
        self.force_params = SamplingParams(temperature=0.0, max_tokens=FORCE_MAX_TOKENS)

    def generate(self, rows):
        """[(prompt_token_ids, token_ids, trace, finish_reason)] for a batch of rows."""
        outs = self.llm.generate([chat_prompt(self.tok, r["problem"]) for r in rows], self.params)
        return [(list(o.prompt_token_ids), list(o.outputs[0].token_ids), o.outputs[0].text,
                 o.outputs[0].finish_reason) for o in outs]

    def force(self, id_lists):
        outs = self.llm.generate([{"prompt_token_ids": ids} for ids in id_lists], self.force_params)
        return [o.outputs[0].text for o in outs]


class MLXBackend:
    def __init__(self, args):
        import importlib
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
        self.mx = mx
        self.generate_step = importlib.import_module("mlx_lm.generate").generate_step
        self.model, self.tok = load(MODEL)  # converts the Hugging Face weights (bf16)
        self.eos = set(self.tok.eos_token_ids)
        self.sampler, self.greedy = make_sampler(temp=0.6, top_p=0.95), make_sampler(temp=0.0)
        self.max_tokens, self.seed = args.max_tokens, args.seed

    def _run(self, ids, max_tokens, sampler):
        out, finish = [], "length"
        for token, _ in self.generate_step(self.mx.array(ids), self.model, max_tokens=max_tokens, sampler=sampler):
            out.append(int(token))
            if out[-1] in self.eos:
                finish = "stop"
                break
        return out, finish

    def generate(self, rows):
        results = []
        for r in rows:
            self.mx.random.seed(self.seed ^ zlib.crc32(r["id"].encode()))  # reproducible per question
            prompt_ids = self.tok.encode(chat_prompt(self.tok, r["problem"]))
            ids, finish = self._run(prompt_ids, self.max_tokens, self.sampler)
            results.append((prompt_ids, ids, decode(self.tok, ids), finish))
        return results

    def force(self, id_lists):
        return [decode(self.tok, self._run(ids, FORCE_MAX_TOKENS, self.greedy)[0]) for ids in id_lists]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def done_ids(path):
    ids = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass  # partial last line from a disconnect
    return ids


def summarize(path):
    rows = [json.loads(l) for l in open(path) if l.strip().endswith("}")]
    if not rows:
        print("no traces written")
        return
    n = len(rows)
    trunc = [r for r in rows if r["truncated"]]
    print(f"--- summary of {path}")
    print(f"questions:        {n}")
    print(f"final accuracy:   {sum(r['final_correct'] for r in rows) / n:.2%}")
    print(f"truncated:        {len(trunc) / n:.2%} ({len(trunc)})")
    if trunc:
        print(f"forced accuracy:  {sum(bool(r['forced_correct']) for r in trunc) / len(trunc):.2%} "
              f"(on truncated traces)")
    print(f"avg tokens:       {sum(r['num_tokens'] for r in rows) / n:.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="JSONL from scripts/sample_inputs.py")
    p.add_argument("--out", required=True, help="output JSONL (appended to; existing IDs skipped)")
    p.add_argument("--backend", choices=["vllm", "mlx"], default="vllm")
    p.add_argument("--n", type=int, default=None, help="only use the first N questions of --input")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--dtype", default="half", choices=["half", "float16", "bfloat16"],
                   help="vLLM only: half for T4; bfloat16 on A100/L4")
    p.add_argument("--batch-size", type=int, default=None,
                   help="questions per generate call, saved after each (default 64 vllm, 1 mlx)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    batch_size = args.batch_size or (64 if args.backend == "vllm" else 1)

    rows = [json.loads(l) for l in open(args.input)]
    if args.n is not None:
        rows = rows[:args.n]
    skip = done_ids(args.out)
    todo = [r for r in rows if r["id"] not in skip]
    print(f"{len(rows)} questions, {len(rows) - len(todo)} already in {args.out}, {len(todo)} to do")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if todo:
        backend = (VLLMBackend if args.backend == "vllm" else MLXBackend)(args)
        t0, n_new_tokens = time.time(), 0
        for b in range(0, len(todo), batch_size):
            batch = todo[b:b + batch_size]
            recs = []
            for r, (prompt_ids, token_ids, trace, finish) in zip(batch, backend.generate(batch)):
                gold = str(r["answer"])
                _, finished, after = trace.partition("</think>")
                final = extract_answer(after) if finished else None  # a cut-off trace has no final answer
                recs.append({
                    "id": r["id"], "dataset": r["dataset"], "split": r.get("split"),
                    "problem": r["problem"], "gold": gold,
                    "prompt_token_ids": prompt_ids, "token_ids": token_ids,
                    "num_tokens": len(token_ids), "finish_reason": finish,
                    "truncated": finish == "length", "trace": trace, "final_answer": final,
                    "final_correct": final is not None and is_correct(final, gold) is True,
                    "forced_answer": None, "forced_correct": None,
                })
                n_new_tokens += len(token_ids)

            # Forced answer for truncated traces: close the thinking and ask for the boxed answer.
            forced = [rec for rec in recs if rec["truncated"]]
            if forced:
                id_lists = [rec["prompt_token_ids"] + rec["token_ids"] +
                            backend.tok.encode(force_suffix(rec["trace"]), add_special_tokens=False) for rec in forced]
                for rec, text in zip(forced, backend.force(id_lists)):
                    ans = last_boxed("\\boxed{" + text)
                    rec["forced_answer"] = ans
                    rec["forced_correct"] = ans is not None and is_correct(ans, rec["gold"]) is True

            with open(args.out, "a") as f:
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")
            done, el = b + len(batch), time.time() - t0
            print(f"[{done}/{len(todo)}] {el / 60:.1f} min, {n_new_tokens / el:.0f} gen tok/s, "
                  f"avg {n_new_tokens / done:.0f} tok/question, "
                  f"ETA {el / done * (len(todo) - done) / 60:.1f} min", flush=True)
        print(f"this session: {len(todo)} questions, {n_new_tokens} tokens in {(time.time() - t0) / 60:.1f} min "
              f"({n_new_tokens / (time.time() - t0):.0f} tok/s)")

    summarize(args.out)


if __name__ == "__main__":
    main()
