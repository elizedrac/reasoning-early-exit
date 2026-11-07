"""
Generate reasoning traces for a small sample of questions, split them into chunks,
and label each chunk's intermediate answer as correct or incorrect.

Usage (on a GPU, e.g. Colab T4):
    pip install vllm math-verify
    python sample_traces.py --input data/gsm8k/train.jsonl --n 100 --out sample_traces.jsonl

Output: one line per question with the full trace and a list of labeled chunks.
"""
import argparse
import json
import random
import re

from math_verify import parse, verify

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
PROMPT = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

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


def chunk_trace(thinking):
    """Split at reflection markers, merging pieces until each chunk contains an answer."""
    pieces = [p for p in MARKERS.split(thinking) if p.strip()]
    chunks, buf = [], ""
    for p in pieces:
        buf += p
        if extract_answer(buf) is not None:
            chunks.append(buf)
            buf = ""
    if buf.strip():
        chunks.append(buf)  # trailing text with no answer
    return chunks


def label(trace, gold):
    thinking, finished, after = trace.partition("</think>")
    out, pos = [], 0
    for k, c in enumerate(chunk_trace(thinking)):
        pos += len(c)
        ans = extract_answer(c)
        out.append({"k": k, "text": c, "end_char": pos, "answer": ans,
                    "correct": is_correct(ans, gold) if ans is not None else None})
    final = extract_answer(after) if finished else None  # a cut-off trace has no final answer
    return out, final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", default="sample_traces.jsonl")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from vllm import LLM, SamplingParams  # imported here so labeling code runs without a GPU

    rows = [json.loads(l) for l in open(args.input)]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.n]

    llm = LLM(model=MODEL, dtype="half", max_model_len=args.max_tokens + 1024)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(problem=r["problem"])}],
        tokenize=False, add_generation_prompt=True) for r in rows]
    params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_tokens, seed=args.seed)
    outputs = llm.generate(prompts, params)

    n_chunks = n_correct_final = 0
    with open(args.out, "w") as f:
        for r, o in zip(rows, outputs):
            trace = o.outputs[0].text
            chunks, final = label(trace, r["answer"])
            final_ok = final is not None and is_correct(final, r["answer"]) is True
            n_chunks += len(chunks)
            n_correct_final += final_ok
            f.write(json.dumps({"id": r["id"], "problem": r["problem"], "gold": r["answer"],
                                "trace": trace, "num_tokens": len(o.outputs[0].token_ids),
                                "final_answer": final, "final_correct": final_ok,
                                "chunks": chunks}) + "\n")
    print(f"{len(rows)} questions, {n_chunks} chunks, final accuracy {n_correct_final / len(rows):.2%}")


if __name__ == "__main__":
    main()
