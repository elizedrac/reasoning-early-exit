"""
Label trace chunks: an open-source LLM extracts intermediate answers, math-verify judges them.

Per trace:
  1. Split the thinking part (before </think>) at every reflection marker ("Wait", "Hmm",
     "Alternatively", ...).
  2. Extract each piece's answer with Qwen2.5-7B-Instruct, one call per piece. The prompt has 4
     worked examples, the question, the last --context-tokens (default 500) tokens of earlier
     reasoning as context, and the current piece, and asks only about the current piece: the
     final answer it states most recently without rejecting it, or NONE. If a reply has no
     ANSWER line, the pilot's regex extractor (generate_traces.extract_answer) is used instead.
     Pieces with no digit or math symbol get NONE without an LLM call (~20% of pieces).
  3. Ground the answers: Unicode math is rewritten as LaTeX (100√3 -> 100\\sqrt{3}), and an answer
     is dropped unless the piece itself writes it (same number, normalized substring, or equal to
     the piece's \\boxed{}). Without this, the extractor repeats an answer from the context for
     pieces that state none.
  4. Merge pieces with no answer into the nearest piece that has one, following Zhang et al.
     (segment until an answer appears): a no-answer piece joins the next answered piece. Pieces
     after the last answer join the last answered chunk. A trace with no answer at all becomes
     one unlabeled chunk.
  5. Judge each chunk's answer against gold with math-verify. Unparseable answers get correct = null.
  6. Map chunks to token spans [start_token, end_token) using the saved token_ids. Records
     without token_ids (pilot traces) get null spans.

Output: input records with `chunks` replaced (fields k, text, start_token, end_token, answer,
correct, n_pieces) plus labeler, piece_answers_raw (extractor output), piece_answers (after
grounding), piece_sources ("llm", "regex", or "skip"). Works with extract_hidden.py.
scripts/relabel.py reruns steps 3 to 6 on existing output without the LLM.

Backends (the same model, 4-bit):
    --backend vllm   Qwen/Qwen2.5-7B-Instruct-AWQ          (NVIDIA; fp16 7B doesn't fit a T4)
    --backend mlx    mlx-community/Qwen2.5-7B-Instruct-4bit (Apple silicon)
On vLLM, whole traces are added to a batch until it holds --batch-chunks pieces (default 2048,
across many traces), and the batch goes out in one call; the shared instructions are a common
prefix, so the prefix cache reuses them. Replies stop at the first newline (at most
MAX_ANSWER_TOKENS tokens). mlx runs one prompt at a time, so use it for small tests only.
Replies are cached in --cache, keyed by model + prompt.

Hand-check (100 reviewed pilot chunks, see pilot/README.md), vLLM on a T4: label accuracy 94% for
the LLM extractor vs. 93% for the regex; extraction accuracy 89% vs. 62%.

Usage:
    python scripts/label_chunks.py --backend mlx --input traces/gsm8k_train.jsonl --out labeled/gsm8k_train.jsonl
    python scripts/label_chunks.py --backend vllm --input traces/gsm8k_train.jsonl --out labeled/gsm8k_train.jsonl
Validate on the 100 reviewed hand-check chunks (regex vs. LLM):
    python scripts/label_chunks.py --handcheck pilot/handcheck.csv --out pilot/handcheck_llm_v2.csv
Resumable: IDs already in --out are skipped.
"""
import argparse
import csv
import fractions
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_traces import (MARKERS, MODEL as GEN_MODEL, add_token_spans, done_ids,  # noqa: E402
                             extract_answer as regex_extract, is_correct, last_boxed)

LABELERS = {"vllm": "Qwen/Qwen2.5-7B-Instruct-AWQ", "mlx": "mlx-community/Qwen2.5-7B-Instruct-4bit"}
MAX_ANSWER_TOKENS = 32  # "ANSWER: <answer>"; the longest MATH gold answers are ~40 tokens, most under 15

SYSTEM = "You read a student's step-by-step math reasoning and report the answer they have reached."

INSTRUCTIONS = """You will see a math question, the student's reasoning so far, and the current excerpt of their reasoning. Report the final answer to the question that the CURRENT EXCERPT states most recently without then rejecting it.
- Only an answer to the question asked counts. Intermediate results, values of helper variables, equations set to zero, digit sums, and numbers restated from the question do not.
- If the current excerpt states no final answer, or rejects the one it states, reply NONE.
- Use the previous reasoning only to understand the excerpt. Don't report an answer that appears only there.
- Write the answer as a bare value or LaTeX expression, without units or words.
Reply with exactly one line: ANSWER: <answer>  or  ANSWER: NONE

Examples:

Question: Mia has 20 marbles and gives 8 to her brother. How many marbles does she have left?
Current excerpt: So she has 20 - 8 = 12 marbles left. Let me verify: 12 + 8 = 20. Yes.
ANSWER: 12

Question: Find the minimum value of x^2 - 6x + 11.
Current excerpt: Wait, I can take the derivative instead: 2x - 6 = 0, so x = 3. Now I need to plug this back in.
ANSWER: NONE

Question: What is the smallest three-digit number whose digits add up to 5?
Current excerpt: Hmm, let me check 113: 1 + 1 + 3 = 5. That also works, but it is larger.
ANSWER: NONE

Question: A triangle has base 8 and height 5. What is its area?
Current excerpt: So the area is 24. Wait, that used the wrong side. With height 5, the area is 1/2 * 8 * 5 = 20.
ANSWER: 20"""

CHUNK_TEMPLATE = INSTRUCTIONS + """

Now the real case.

Question:
{problem}

Previous reasoning (context only):
\"\"\"{context}\"\"\"

Current excerpt:
\"\"\"{chunk}\"\"\"

What final answer does the current excerpt state? Reply with exactly one line: ANSWER: <answer>  or  ANSWER: NONE"""

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


SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
VULGAR = {"½": "\\frac{1}{2}", "⅓": "\\frac{1}{3}", "⅔": "\\frac{2}{3}", "¼": "\\frac{1}{4}", "¾": "\\frac{3}{4}"}
UNICODE_MATH = {"−": "-", "–": "-", "×": "\\times ", "·": "\\cdot ", "÷": "/", "π": "\\pi ", "∞": "\\infty ",
                "≤": "\\le ", "≥": "\\ge ", "≠": "\\ne ", "°": "^\\circ"}


def to_latex(s):
    """Unicode math (100√3, x², ½, π) as LaTeX, so math-verify can parse it."""
    for u, l in {**UNICODE_MATH, **VULGAR}.items():
        s = s.replace(u, l)
    s = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+", lambda m: "^{" + m.group().translate(SUPERSCRIPT) + "}", s)
    s = re.sub(r"√\s*\(([^()]*)\)", r"\\sqrt{\1}", s)
    s = re.sub(r"√\s*(\{[^{}]*\})", r"\\sqrt\1", s)
    s = re.sub(r"√\s*([A-Za-z0-9.]+)", r"\\sqrt{\1}", s)
    return s


# A piece with none of these can't state an answer we could ground, so it gets NONE without an LLM call.
MATHY = re.compile(r"[0-9\\√⁰¹²³⁴⁵⁶⁷⁸⁹½⅓⅔¼¾π∞]")


def squash_math(s):
    """Normalize LaTeX/plain math text for substring matching."""
    s = to_latex(s)
    s = re.sub(r"(?<!\\)\bsqrt\s*\(?\s*([A-Za-z0-9.]+)\s*\)?", r"\\sqrt{\1}", s)
    s = re.sub(r"\\(?:left|right|displaystyle|[,;!: ])", "", s)
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    s = re.sub(r"[\s$]", "", s)
    s = re.sub(r"\^\{(\w)\}", r"^\1", s)
    return s


NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?|\\frac\{-?\d+\}\{\d+\}")


def as_number(s):
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(\d+)\}", s)
    try:
        return fractions.Fraction(int(m.group(1)), int(m.group(2))) if m else fractions.Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def grounded(ans, text):
    """True if the answer is actually written in the piece (not only in the context or computed by the
    extractor): as a normalized substring, as the same number, or math-verify-equal to the piece's \\boxed{}."""
    a, t = squash_math(ans), squash_math(text)
    if not a:
        return False
    value = as_number(a.rstrip("%"))
    if value is not None:
        return any(as_number(n) == value for n in NUM.findall(t))
    if a in t:
        return True
    boxed = last_boxed(text)
    return bool(boxed) and is_correct(boxed, ans) is True


def ground_answers(texts, answers):
    """Drop answers that the piece doesn't state itself (see grounded)."""
    return [a if a is not None and MATHY.search(t) and grounded(a, t) else None for t, a in zip(texts, answers)]


def clean(ans):
    if ans is None:
        return None
    ans = to_latex(str(ans).strip())
    if "\\boxed" in ans:
        ans = last_boxed(ans) or ans
    ans = re.sub(r"\\[()\[\]]", "", ans).replace("$", "").strip().rstrip(".").strip()
    if not ans or ans.upper().strip("*`'\" ") in ("NONE", "NULL", "N/A"):
        return None
    return ans


def parse_reply(text):
    """'ANSWER: x' -> ('x', True); 'ANSWER: NONE' -> (None, True); no ANSWER line -> (None, False)."""
    m = ANSWER_LINE.search(text)
    if not m:
        return None, False
    return clean(m.group(1).strip().splitlines()[0] if m.group(1).strip() else ""), True


def tail_tokens(tok, text, n):
    """The last n tokens of text (as text), prefixed with "[...] " if anything was cut."""
    if n <= 0:
        return "[...] " if text.strip() else ""
    window = text[-8 * n:]  # a token is rarely over 8 chars; avoids tokenizing whole traces
    ids = tok.encode(window, add_special_tokens=False)
    if len(ids) <= n and len(window) == len(text):
        return text
    return "[...] " + tok.decode(ids[-n:])


def chunk_prompt(tok, problem, context, chunk, context_tokens, max_chunk_tokens):
    context = tail_tokens(tok, context, context_tokens)
    chunk = tail_tokens(tok, chunk, max_chunk_tokens)  # guard for runaway pieces; answers come late
    user = CHUNK_TEMPLATE.format(problem=problem.strip(), context=context.strip() or "(none, this is the start)",
                                 chunk=chunk.strip())
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


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
            self.llm = LLM(model=model, dtype="half", max_model_len=max_model_len, max_num_seqs=256,
                           gpu_memory_utilization=gpu_mem, enable_prefix_caching=True, seed=0)
            self.tok = self.llm.get_tokenizer()
            self.params = SamplingParams(temperature=0.0, max_tokens=MAX_ANSWER_TOKENS, stop=["\n"])
        else:
            from mlx_lm import load
            from mlx_lm.sample_utils import make_sampler
            self.llm, self.tok = load(model)
            self.sampler = make_sampler(temp=0.0)

    def generate(self, prompts):
        """Greedy replies for prompts (list of str), using and filling the cache."""
        keys = [hashlib.sha256(f"{self.model}\0{MAX_ANSWER_TOKENS}\0{p}".encode()).hexdigest() for p in prompts]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        if todo:
            t0 = time.time()
            if self.backend == "vllm":
                # One batched call; vLLM schedules all prompts together and shares cached prefixes.
                outs = [o.outputs[0].text for o in self.llm.generate([prompts[i] for i in todo], self.params,
                                                                     use_tqdm=False)]
                self._save([(keys[i], out) for i, out in zip(todo, outs)])
            else:
                from mlx_lm import generate
                for i in todo:  # sequential; save each reply as soon as it is done
                    self._save([(keys[i], generate(self.llm, self.tok, prompt=prompts[i], max_tokens=MAX_ANSWER_TOKENS,
                                                   sampler=self.sampler, verbose=False))])
            self.llm_seconds += time.time() - t0
            self.n_llm_calls += len(todo)
        return [self.cache[k] for k in keys]

    def _save(self, pairs):
        with open(self.cache_path, "a") as f:
            for key, out in pairs:
                self.cache[key] = out
                f.write(json.dumps({"key": key, "reply": out}) + "\n")


def piece_prompts(ex, problem, texts, args):
    """One prompt per piece that could state an answer (MATHY), each with the tail of the earlier
    pieces as context. Returns (prompts, indices of the pieces they belong to)."""
    prompts, idx, context = [], [], ""
    for i, t in enumerate(texts):
        if MATHY.search(t):
            prompts.append(chunk_prompt(ex.tok, problem, context, t, args.context_tokens, args.max_chunk_tokens))
            idx.append(i)
        context += t
    return prompts, idx


def answers_from_replies(texts, replies, idx=None):
    """(answer or None, source) per piece; the regex extractor for a reply with no ANSWER line.
    Pieces not in idx (no LLM call) get (None, "skip")."""
    idx = list(range(len(texts))) if idx is None else idx
    out = [(None, "skip")] * len(texts)
    for i, reply in zip(idx, replies):
        ans, ok = parse_reply(reply)
        out[i] = (ans, "llm") if ok else (regex_extract(texts[i]), "regex")
    return out


def build_record(rec, raw_answers, sources, labeler, gen_tok, log):
    """Ground the piece answers, merge pieces into chunks, judge them, and add token spans.
    Returns (output record, number of token-span mismatches)."""
    thinking = rec["trace"].partition("</think>")[0]
    spans = split_pieces(thinking)
    texts = [thinking[x:y] for x, y in spans]
    assert len(texts) == len(raw_answers), f"{rec['id']}: {len(texts)} pieces but {len(raw_answers)} answers"
    raw_answers = [clean(a) for a in raw_answers]
    answers = ground_answers(texts, raw_answers)
    chunks = [{"k": k, "text": thinking[x:y], "start_char": x, "end_char": y, "answer": ans,
               "correct": is_correct(ans, str(rec["gold"])) if ans is not None else None, "n_pieces": n}
              for k, (x, y, ans, n) in enumerate(merge_pieces(spans, answers))]
    n_bad = 0
    if rec.get("token_ids"):
        n_bad = add_token_spans(gen_tok, rec["trace"], rec["token_ids"], chunks, rec["id"], log)
    else:
        for c in chunks:
            c["start_token"] = c["end_token"] = None
    out = dict(rec, labeler=labeler, piece_answers_raw=raw_answers, piece_answers=answers, piece_sources=sources,
               chunks=[{key: c[key] for key in ("k", "text", "start_token", "end_token", "answer", "correct",
                                                "n_pieces")} for c in chunks])
    return out, n_bad


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
    n_mismatch, t0 = 0, time.time()

    def log(entry):
        with open(mismatch_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    batch_chunks = args.batch_chunks or (2048 if args.backend == "vllm" else 1)
    i = 0
    while i < len(todo):
        # Add whole traces until the batch holds batch_chunks pieces, then send all of them in one call.
        prepared, prompts = [], []
        while i < len(todo) and (not prompts or len(prompts) < batch_chunks):
            rec = todo[i]
            i += 1
            thinking = rec["trace"].partition("</think>")[0]
            texts = [thinking[x:y] for x, y in split_pieces(thinking)]
            ps, idx = piece_prompts(ex, rec["problem"], texts, args)
            prepared.append((rec, texts, idx, len(prompts), len(prompts) + len(ps)))
            prompts += ps
        replies = ex.generate(prompts)

        with open(args.out, "a") as f:
            for rec, texts, idx, lo, hi in prepared:
                extracted = answers_from_replies(texts, replies[lo:hi], idx)
                out, bad = build_record(rec, [a for a, _ in extracted], [s for _, s in extracted], ex.model,
                                        gen_tok, log)
                n_mismatch += bad
                f.write(json.dumps(out) + "\n")
        if len(prepared) > 1 or i % args.log_every == 0 or i == len(todo):
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] {el / 60:.1f} min, {el / i:.1f} s/trace, "
                  f"est. {el / i * (len(todo) - i) / 60:.1f} min left", flush=True)

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
        s = stats.setdefault(dataset_of(r), {"traces": 0, "pieces": 0, "chunks": 0, "labeled": 0, "pos": 0, "regex": 0})
        s["traces"] += 1
        s["pieces"] += len(r["piece_answers"])
        s["chunks"] += len(r["chunks"])
        s["labeled"] += sum(c["correct"] is not None for c in r["chunks"])
        s["pos"] += sum(c["correct"] is True for c in r["chunks"])
        s["regex"] += sum(x == "regex" for x in r["piece_sources"])
        s["skip"] = s.get("skip", 0) + sum(x == "skip" for x in r["piece_sources"])
    print(f"--- summary of {path}")
    for d, s in stats.items():
        n = s["traces"]
        print(f"{d:9s} traces {n:5d}  pieces/trace {s['pieces'] / n:5.1f}  chunks/trace {s['chunks'] / n:4.1f}  "
              f"labeled/trace {s['labeled'] / n:4.1f}  positive {s['pos'] / max(s['labeled'], 1):.0%}  "
              f"regex fallback {s['regex'] / max(s['pieces'], 1):.1%}  no LLM call {s['skip'] / max(s['pieces'], 1):.0%} of pieces")


def same_answer(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.strip() == b.strip() or is_correct(a, b) is True


def handcheck(args, ex):
    """Score the regex and the LLM extractor on the 100 reviewed hand-check chunks (the pilot's regex
    chunks; the LLM sees the tail of the earlier chunks of the trace as context).

    Truth: `true_answer` (blank = the chunk states no final answer).
    extraction: extracted answer equals the true answer (both none counts as equal)
    label:      correct vs. not, where no answer counts as not correct
    label3:     correct / incorrect / no answer
    """
    from make_handcheck import SOURCES
    traces = {}
    for path in SOURCES.values():
        for line in open(path):
            d = json.loads(line)
            traces[d["id"]] = d
    rows = list(csv.DictReader(open(args.handcheck)))
    assert "true_answer" in rows[0], "hand-check CSV needs a true_answer column"

    prompts = []
    for r in rows:
        d, k = traces[r["id"]], int(r["k"])
        context = "".join(c["text"] for c in d["chunks"][:k])
        prompts.append(chunk_prompt(ex.tok, d["problem"], context, d["chunks"][k]["text"],
                                    args.context_tokens, args.max_chunk_tokens))
    t0 = time.time()
    replies = ex.generate(prompts)
    print(f"{ex.n_llm_calls} new LLM calls in {(time.time() - t0) / 60:.1f} min (the rest were cached)")

    def label3(ans, gold):
        return None if ans is None else int(is_correct(ans, gold) is True)

    systems = ["regex", "LLM"]
    score, out = {}, []
    for r, reply in zip(rows, replies):
        d, k = traces[r["id"]], int(r["k"])
        text = d["chunks"][k]["text"]
        truth = r["true_answer"].strip() or None
        (llm, src), = answers_from_replies([text], [reply])
        llm, = ground_answers([text], [llm])
        preds = {"regex": regex_extract(text), "LLM": llm}
        t3 = label3(truth, r["gold"])
        row = {"id": r["id"], "dataset": r["dataset"], "k": r["k"], "gold": r["gold"], "true_answer": truth or "",
               "llm_source": src, "review_notes": r["notes"]}
        for name in systems:
            p3 = label3(preds[name], r["gold"])
            res = {"extraction": int(same_answer(preds[name], truth)),
                   "label": int(bool(p3) == bool(t3)), "label3": int(p3 == t3)}
            row[f"{name} answer"] = preds[name] or ""
            for m, v in res.items():
                row[f"{name} {m}"] = v
                for key in (r["dataset"], "all"):
                    score.setdefault((key, name, m), []).append(v)
        row["llm_raw"] = reply.strip()
        out.append(row)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    for m in ("extraction", "label", "label3"):
        print(f"\n{m:10s} " + "".join(f"{s:>8s}" for s in systems))
        for key in ("all", "gsm8k", "math"):
            print(f"{key:10s} " + "".join(f"{sum(score[(key, s, m)]) / len(score[(key, s, m)]):8.0%}" for s in systems))
    print("\nlabel: no answer counts as not correct; label3: correct / incorrect / no answer")
    print(f"LLM replies without an ANSWER line (regex used): {sum(r['llm_source'] == 'regex' for r in out)}")
    misses = [(r["id"], r["k"], r["true_answer"] or "NONE", r["LLM answer"] or "NONE") for r in out
              if not r["LLM extraction"]]
    print(f"LLM extraction misses (id, k, truth, llm): {misses}")
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
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--context-tokens", type=int, default=500,
                   help="earlier reasoning shown as context: its last N tokens")
    p.add_argument("--max-chunk-tokens", type=int, default=6000,
                   help="a longer piece keeps only its last N tokens (keeps prompts under --max-model-len)")
    p.add_argument("--batch-chunks", type=int, default=None,
                   help="pieces (across traces) sent to the extractor in one call (default 2048 vllm, 1 mlx)")
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
