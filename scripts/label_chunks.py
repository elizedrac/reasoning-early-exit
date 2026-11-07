"""
Label trace chunks: an open-source LLM extracts intermediate answers, math-verify judges them.

Replaces the regex extraction of the pilot, which the hand-check found unreliable (62% of
extracted answers were what the chunk commits to; 40% on MATH).

Per trace:
  1. Split the thinking part (before </think>) at every reflection marker ("Wait", "Hmm",
     "Alternatively", ...). No regex-based merging.
  2. Extract answers with Qwen2.5-7B-Instruct, one call per trace: the prompt lists the numbered
     pieces and asks for a JSON list with one entry per piece (the final answer the piece states
     most recently without rejecting it, or null). If the reply does not have exactly one entry
     per piece, or the trace is too long for one call, fall back to one call per piece.
  3. Merge pieces with no answer into the nearest piece that has one, following Zhang et al.
     (segment until an answer appears): a no-answer piece joins the next answered piece. Pieces
     after the last answer join the last answered chunk. A trace with no answer at all becomes
     one unlabeled chunk.
  4. Judge each chunk's answer against gold with math-verify (is_correct from generate_traces).
     Answers math-verify can't parse get correct = null.
  5. Map chunks to token spans [start_token, end_token) using the saved token_ids. Records
     without token_ids (pilot traces) get null spans.

Output: input records with `chunks` replaced (fields k, text, start_token, end_token, answer,
correct, n_pieces) plus labeler, label_mode ("trace" or "chunk"), piece_answers. Works with
extract_hidden.py.

Backends (the same model, 4-bit):
    --backend vllm   Qwen/Qwen2.5-7B-Instruct-AWQ          (NVIDIA; fp16 7B doesn't fit a T4)
    --backend mlx    mlx-community/Qwen2.5-7B-Instruct-4bit (Apple silicon)
Replies are cached in --cache, keyed by model + prompt, so reruns and resumes are free.

Usage:
    python scripts/label_chunks.py --backend mlx --input traces/gsm8k_train.jsonl --out labeled/gsm8k_train.jsonl
Validate against the manual hand-check (labels the pilot's regex chunks, so the 100 reviewed
chunks line up exactly):
    python scripts/label_chunks.py --backend mlx --handcheck pilot/handcheck.csv --out pilot/handcheck_llm.csv
Resumable: IDs already in --out are skipped.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_traces import MARKERS, MODEL as GEN_MODEL, add_token_spans, done_ids, is_correct, last_boxed  # noqa: E402

LABELERS = {"vllm": "Qwen/Qwen2.5-7B-Instruct-AWQ", "mlx": "mlx-community/Qwen2.5-7B-Instruct-4bit"}
CONTEXT_CHARS = 300

SYSTEM = "You read a student's step-by-step math reasoning and report the answers they reach."
RULES = """Rules:
- Only an answer to the question the problem asks. Ignore intermediate quantities, values of helper variables, and numbers restated from the problem.
- If a segment states no final answer, or rejects or abandons the one it states, that segment has none.
- Write answers as a bare value or LaTeX expression, without units or words."""

TRACE_TEMPLATE = """Problem:
{problem}

The student's reasoning, split into {n} numbered segments:

{segments}

For each segment, which final answer to the problem does that segment state most recently without then rejecting it?
""" + RULES + """

Reply with only a JSON list of exactly {n} entries, one per segment in order: the answer as a string, or null if the segment states none. Example for 3 segments: [null, "12", null]"""

CHUNK_TEMPLATE = """Problem:
{problem}

End of the preceding reasoning (context only):
\"\"\"{context}\"\"\"

Excerpt:
\"\"\"{chunk}\"\"\"

Which final answer to the problem does the excerpt state most recently without then rejecting it?
""" + RULES + """

Reply with exactly one line: ANSWER: <answer>  or  ANSWER: NONE"""

ANSWER_LINE = re.compile(r"ANSWER:\s*(.+)", re.I)


# ---------------------------------------------------------------------------
# Chunking, prompts, parsing (CPU)
# ---------------------------------------------------------------------------

def split_pieces(thinking):
    """[(start_char, end_char)] split at every marker, contiguous over the whole text.
    A whitespace-only leading piece is joined to the next one."""
    cuts = sorted({0, len(thinking)} | {m.start() for m in MARKERS.finditer(thinking)})
    spans = [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]
    if len(spans) > 1 and not thinking[spans[0][0]:spans[0][1]].strip():
        spans[1] = (spans[0][0], spans[1][1])
        spans = spans[1:]
    return spans


def merge_pieces(spans, answers):
    """Merge no-answer pieces into the nearest answered piece (forward; trailing ones backward).
    Returns [(start_char, end_char, answer, n_pieces)]."""
    groups, start, n = [], None, 0
    for (a, b), ans in zip(spans, answers):
        start = a if start is None else start
        n += 1
        if ans is not None:
            groups.append([start, b, ans, n])
            start, n = None, 0
    if start is not None:  # pieces after the last answer
        if groups:
            groups[-1][1] = spans[-1][1]
            groups[-1][3] += n
        else:
            groups.append([start, spans[-1][1], None, n])
    return [tuple(g) for g in groups]


def clean(ans):
    if ans is None:
        return None
    ans = str(ans).strip()
    if "\\boxed" in ans:
        ans = last_boxed(ans) or ans
    ans = ans.replace("$", "").strip().rstrip(".").strip()
    if not ans or ans.upper().strip("*`'\" ") in ("NONE", "NULL", "N/A"):
        return None
    return ans


def parse_chunk_reply(text):
    m = ANSWER_LINE.search(text)
    return clean(m.group(1).strip().splitlines()[0]) if m else None


def parse_trace_reply(text, n):
    """JSON list of n answers, or None if the reply isn't one."""
    i, j = text.find("["), text.rfind("]")
    if i == -1 or j <= i:
        return None
    body = text[i:j + 1]
    # LaTeX backslashes are usually not valid JSON escapes (\f in \frac is a form feed): escape them all.
    body = body.replace("\\\\", "\\").replace("\\", "\\\\")
    try:
        items = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list) or len(items) != n:
        return None
    return [clean(x) if isinstance(x, (str, int, float)) else None for x in items]


def chat(tok, user):
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def trace_prompt(tok, problem, texts):
    segments = "\n\n".join(f"[{i + 1}]\n{t.strip()}" for i, t in enumerate(texts))
    return chat(tok, TRACE_TEMPLATE.format(problem=problem.strip(), n=len(texts), segments=segments))


def chunk_prompt(tok, problem, context, chunk, max_chars):
    if len(chunk) > max_chars:
        chunk = "[...] " + chunk[-max_chars:]
    context = context[-CONTEXT_CHARS:].strip() or "(start of reasoning)"
    return chat(tok, CHUNK_TEMPLATE.format(problem=problem.strip(), context=context, chunk=chunk.strip()))


# ---------------------------------------------------------------------------
# Extractor backends with a reply cache
# ---------------------------------------------------------------------------

class Extractor:
    def __init__(self, backend, model, cache_path, max_model_len, gpu_mem):
        self.backend, self.model, self.cache_path = backend, model, cache_path
        self.cache, self.n_llm_calls, self.llm_seconds = {}, 0, 0.0
        if cache_path and os.path.exists(cache_path):
            for line in open(cache_path):
                try:
                    e = json.loads(line)
                    self.cache[e["key"]] = e["reply"]
                except (json.JSONDecodeError, KeyError):
                    pass
        if backend == "vllm":
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=model, dtype="half", max_model_len=max_model_len,
                           gpu_memory_utilization=gpu_mem, enable_prefix_caching=True, seed=0)
            self.tok = self.llm.get_tokenizer()
            self._params = lambda m: SamplingParams(temperature=0.0, max_tokens=m)
        else:
            from mlx_lm import load
            from mlx_lm.sample_utils import make_sampler
            self.llm, self.tok = load(model)
            self.sampler = make_sampler(temp=0.0)
        self.max_model_len = max_model_len

    def n_tokens(self, prompt):
        return len(self.tok.encode(prompt))

    def generate(self, prompts, max_tokens):
        """Greedy replies for prompts (list of str), using and filling the cache."""
        keys = [hashlib.sha256(f"{self.model}\0{max_tokens}\0{p}".encode()).hexdigest() for p in prompts]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        if todo:
            t0 = time.time()
            if self.backend == "vllm":
                outs = [o.outputs[0].text for o in self.llm.generate([prompts[i] for i in todo],
                                                                     self._params(max_tokens), use_tqdm=False)]
            else:
                from mlx_lm import generate
                outs = [generate(self.llm, self.tok, prompt=prompts[i], max_tokens=max_tokens,
                                 sampler=self.sampler, verbose=False) for i in todo]
            self.llm_seconds += time.time() - t0
            self.n_llm_calls += len(todo)
            with open(self.cache_path, "a") as f:
                for i, out in zip(todo, outs):
                    self.cache[keys[i]] = out
                    f.write(json.dumps({"key": keys[i], "reply": out}) + "\n")
        return [self.cache[k] for k in keys]


def extract_answers(ex, problem, texts, args):
    """One answer (or None) per text. Tries one call per trace, falls back to one per chunk.
    Returns (answers, mode)."""
    if not texts:
        return [], "trace"
    if len(texts) <= args.max_pieces_per_call:
        prompt = trace_prompt(ex.tok, problem, texts)
        out_budget = min(16 * len(texts) + 64, 4096)
        if ex.n_tokens(prompt) + out_budget <= min(args.max_trace_tokens, ex.max_model_len):
            answers = parse_trace_reply(ex.generate([prompt], out_budget)[0], len(texts))
            if answers is not None:
                return answers, "trace"
    context = ""
    prompts = []
    for t in texts:
        prompts.append(chunk_prompt(ex.tok, problem, context, t, args.max_chunk_chars))
        context += t
    return [parse_chunk_reply(r) for r in ex.generate(prompts, 48)], "chunk"


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def dataset_of(rec):
    return rec.get("dataset") or rec["id"].split("-")[0]


def label_traces(args, ex):
    from transformers import AutoTokenizer
    gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL)  # tokenizer only, for token spans

    rows = [json.loads(l) for l in open(args.input)]
    if args.n is not None:
        rows = rows[:args.n]
    skip = done_ids(args.out)
    todo = [r for r in rows if r["id"] not in skip]
    print(f"{len(rows)} traces, {len(rows) - len(todo)} already in {args.out}, {len(todo)} to do")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mismatch_path = args.out + ".mismatches.jsonl"
    n_mismatch, n_fallback, t0 = 0, 0, time.time()

    def log(entry):
        with open(mismatch_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    for i, rec in enumerate(todo, 1):
        thinking = rec["trace"].partition("</think>")[0]
        spans = split_pieces(thinking)
        answers, mode = extract_answers(ex, rec["problem"], [thinking[a:b] for a, b in spans], args)
        n_fallback += mode == "chunk"
        chunks = [{"k": k, "text": thinking[a:b], "start_char": a, "end_char": b, "answer": ans,
                   "correct": is_correct(ans, str(rec["gold"])) if ans is not None else None, "n_pieces": n}
                  for k, (a, b, ans, n) in enumerate(merge_pieces(spans, answers))]
        if rec.get("token_ids"):
            n_mismatch += add_token_spans(gen_tok, rec["trace"], rec["token_ids"], chunks, rec["id"], log)
        else:
            for c in chunks:
                c["start_token"] = c["end_token"] = None
        rec = dict(rec, labeler=ex.model, label_mode=mode, piece_answers=answers, chunks=[
            {key: c[key] for key in ("k", "text", "start_token", "end_token", "answer", "correct", "n_pieces")}
            for c in chunks])
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if i % args.log_every == 0 or i == len(todo):
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] {el / 60:.1f} min, {el / i:.1f} s/trace, "
                  f"est. {el / i * (len(todo) - i) / 60:.1f} min left, {n_fallback} per-chunk fallbacks", flush=True)

    if todo:
        print(f"extractor: {ex.n_llm_calls} LLM calls, {ex.llm_seconds:.0f} s "
              f"({ex.llm_seconds / len(todo):.1f} s/trace, cache hits excluded)")
    if n_mismatch:
        print(f"WARNING: {n_mismatch} chunk token spans did not match their text; see {mismatch_path}")
    summarize(args.out)


def summarize(path):
    rows = [json.loads(l) for l in open(path) if l.strip().endswith("}")]
    stats = {}
    for r in rows:
        s = stats.setdefault(dataset_of(r), {"traces": 0, "pieces": 0, "chunks": 0, "labeled": 0, "pos": 0, "fallback": 0})
        s["traces"] += 1
        s["pieces"] += len(r["piece_answers"])
        s["chunks"] += len(r["chunks"])
        s["labeled"] += sum(c["correct"] is not None for c in r["chunks"])
        s["pos"] += sum(c["correct"] is True for c in r["chunks"])
        s["fallback"] += r["label_mode"] == "chunk"
    print(f"--- summary of {path}")
    for d, s in stats.items():
        n = s["traces"]
        print(f"{d:9s} traces {n:5d}  pieces/trace {s['pieces'] / n:5.1f}  chunks/trace {s['chunks'] / n:4.1f}  "
              f"labeled/trace {s['labeled'] / n:4.1f}  positive {s['pos'] / max(s['labeled'], 1):.0%}  "
              f"per-chunk fallback {s['fallback'] / n:.0%}")


def same_answer(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.strip() == b.strip() or is_correct(a, b) is True


def handcheck(args, ex):
    """Extract answers for the pilot's regex chunks of each reviewed trace (so the reviewed chunk is
    one of the segments) and score regex vs. LLM against the reviewed true answers.

    Truth: `true_answer` (blank = the chunk states no final answer).
    Extraction accuracy: extracted answer equals the true answer (both blank counts as equal).
    Label accuracy: label 1 iff the answer is correct vs. gold; no answer counts as 0.
    """
    from make_handcheck import SOURCES
    traces = {}
    for path in SOURCES.values():
        for line in open(path):
            d = json.loads(line)
            traces[d["id"]] = d
    rows = list(csv.DictReader(open(args.handcheck)))
    assert "true_answer" in rows[0], "hand-check CSV needs a true_answer column"

    llm_by_trace, modes = {}, {}
    t0 = time.time()
    for tid in dict.fromkeys(r["id"] for r in rows):
        d = traces[tid]
        llm_by_trace[tid], modes[tid] = extract_answers(ex, d["problem"], [c["text"] for c in d["chunks"]], args)
    print(f"extracted {len(llm_by_trace)} traces in {(time.time() - t0) / 60:.1f} min, "
          f"{sum(m == 'chunk' for m in modes.values())} per-chunk fallbacks")

    out, score = [], {}
    for r in rows:
        truth = r["true_answer"].strip() or None
        truth_label = int(truth is not None and is_correct(truth, r["gold"]) is True)
        llm = llm_by_trace[r["id"]][int(r["k"])]
        llm_label = int(llm is not None and is_correct(llm, r["gold"]) is True)
        res = {"regex_extract": int(same_answer(r["extracted"], truth)), "regex_label": int(int(r["label"]) == truth_label),
               "llm_extract": int(same_answer(llm, truth)), "llm_label": int(llm_label == truth_label)}
        for key in (r["dataset"], "all"):
            for m, v in res.items():
                score.setdefault(key, {}).setdefault(m, []).append(v)
        out.append({"id": r["id"], "dataset": r["dataset"], "k": r["k"], "gold": r["gold"],
                    "true_answer": truth or "", "regex_answer": r["extracted"], "llm_answer": llm or "",
                    "truth_label": truth_label, "llm_label_value": llm_label, **res,
                    "label_mode": modes[r["id"]], "review_notes": r["notes"]})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    print(f"{'':7s}{'n':>4s}  {'extraction regex':>17s} {'LLM':>5s}   {'label regex':>12s} {'LLM':>5s}")
    for key in ("all", "gsm8k", "math"):
        s = score[key]
        pct = {m: sum(v) / len(v) for m, v in s.items()}
        print(f"{key:7s}{len(s['llm_label']):4d}  {pct['regex_extract']:17.0%} {pct['llm_extract']:5.0%}   "
              f"{pct['regex_label']:12.0%} {pct['llm_label']:5.0%}")
    bad = [(r["id"], r["k"], r["true_answer"] or "NONE", r["llm_answer"] or "NONE") for r in out if not r["llm_extract"]]
    print(f"LLM extraction misses (id, k, truth, llm): {bad}")
    print(f"wrote {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="traces JSONL from generate_traces.py (or pilot traces)")
    p.add_argument("--out", required=True)
    p.add_argument("--handcheck", metavar="CSV", help="validate on the reviewed hand-check CSV instead")
    p.add_argument("--backend", choices=["vllm", "mlx"], default="vllm")
    p.add_argument("--model", default=None, help="default: the 4-bit Qwen2.5-7B-Instruct for the backend")
    p.add_argument("--cache", default="cache/label_cache.jsonl")
    p.add_argument("--n", type=int, default=None, help="only use the first N traces")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-trace-tokens", type=int, default=24000,
                   help="per-trace prompt + reply budget above which we use per-chunk calls")
    p.add_argument("--max-pieces-per-call", type=int, default=120)
    p.add_argument("--max-chunk-chars", type=int, default=3000, help="per-chunk prompts keep the last N chars")
    p.add_argument("--gpu-mem", type=float, default=0.9, help="vLLM gpu_memory_utilization")
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    model = args.model or LABELERS[args.backend]
    os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
    ex = Extractor(args.backend, model, args.cache, args.max_model_len, args.gpu_mem)
    if args.handcheck:
        handcheck(args, ex)
    else:
        assert args.input, "--input is required unless --handcheck is given"
        label_traces(args, ex)


if __name__ == "__main__":
    main()
