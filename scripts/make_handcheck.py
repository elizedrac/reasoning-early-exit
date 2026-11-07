"""
Sample labeled chunks from the pilot traces for a manual check of the regex + math-verify labels.

Samples 100 chunks with a non-null label (seed 0): 50 from GSM8K and 50 from MATH. Writes a CSV
with the end of each chunk, the extracted answer, the gold answer, and the automatic label, plus
empty review columns:
    true_answer     the final answer the chunk states most recently without rejecting it (blank = none)
    extract_ok      1 if the extracted answer is what the chunk states, else 0
    label_ok        1 if the automatic correct/incorrect label is right, else 0
    notes

Usage:
    python scripts/make_handcheck.py --out pilot/handcheck.csv
    # fill in the review columns, then:
    python scripts/make_handcheck.py --summarize pilot/handcheck.csv

If label accuracy is below 95%, consider labeling with a model verifier instead of pure regex.
"""
import argparse
import csv
import json
import random

SOURCES = {"gsm8k": "pilot/gsm8k_sample.jsonl", "math": "pilot/math_sample.jsonl"}
REVIEW_COLS = ["true_answer", "extract_ok", "label_ok", "notes"]
TAIL_CHARS = 600


def sample(n, seed):
    rng = random.Random(seed)
    out = []
    per_source = [n // 2, n - n // 2]
    for (dataset, path), m in zip(SOURCES.items(), per_source):
        chunks = []
        for line in open(path):
            d = json.loads(line)
            for c in d["chunks"]:
                if c["correct"] is not None:
                    chunks.append({"id": d["id"], "dataset": dataset, "k": c["k"],
                                   "num_chunks": len(d["chunks"]), "gold": str(d["gold"]),
                                   "extracted": c["answer"], "label": int(c["correct"]),
                                   "chunk_tail": c["text"][-TAIL_CHARS:]})
        out += rng.sample(chunks, m)
    return out


def summarize(path, threshold=0.95):
    rows = [r for r in csv.DictReader(open(path)) if r["label_ok"].strip()]
    if not rows:
        print("no reviewed rows")
        return
    for name, subset in [("all", rows)] + [(d, [r for r in rows if r["dataset"] == d]) for d in SOURCES]:
        if not subset:
            continue
        ext = sum(int(r["extract_ok"]) for r in subset) / len(subset)
        lab = sum(int(r["label_ok"]) for r in subset) / len(subset)
        pos = sum(int(r["label"]) for r in subset) / len(subset)
        print(f"{name:6s} n={len(subset):3d}  extraction ok {ext:.0%}  label ok {lab:.0%}  "
              f"(auto label positive rate {pos:.0%})")
    lab = sum(int(r["label_ok"]) for r in rows) / len(rows)
    if lab < threshold:
        print(f"label accuracy {lab:.0%} < {threshold:.0%}: consider a model verifier instead of pure regex")
    else:
        print(f"label accuracy {lab:.0%} >= {threshold:.0%}: regex + math-verify labels look good enough")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="pilot/handcheck.csv")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summarize", metavar="CSV", help="score a reviewed CSV instead of sampling")
    args = p.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return
    rows = sample(args.n, args.seed)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + REVIEW_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({**r, **{c: "" for c in REVIEW_COLS}})
    print(f"wrote {len(rows)} chunks to {args.out}")


if __name__ == "__main__":
    main()
