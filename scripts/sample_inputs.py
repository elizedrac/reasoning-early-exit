"""
Build the question sets used for trace generation.

    train: 1,000 random questions each from GSM8K train and MATH train
    dev:   200 random questions each from GSM8K dev and MATH dev
    test:  all of GSM8K test (1,319), MATH-500 (500), and AIME 2024 (30)

Sampling uses seed 0. The script checks that no question (by ID or by problem text)
appears in more than one output file.

Output: data/trace_inputs/{dataset}_{split}.jsonl, where {dataset} is the record's
`dataset` field (gsm8k, math, math500, aime2024). Each record gets a `split` field.

Usage (from the repo root):
    python scripts/sample_inputs.py
"""
import argparse
import json
import random
from pathlib import Path

# (source file, split, number to sample or None for all)
PLAN = [
    ("gsm8k/train.jsonl", "train", 1000),
    ("math/train.jsonl", "train", 1000),
    ("gsm8k/dev.jsonl", "dev", 200),
    ("math/dev.jsonl", "dev", 200),
    ("gsm8k/test.jsonl", "test", None),
    ("math/test.jsonl", "test", None),
    ("aime/test.jsonl", "test", None),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="data/trace_inputs")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_ids, seen_problems = {}, {}

    for src, split, n in PLAN:
        rows = [json.loads(l) for l in open(Path(args.data) / src)]
        if n is not None:
            rows = random.Random(args.seed).sample(rows, n)
        dataset = rows[0]["dataset"]
        name = f"{dataset}_{split}"
        for r in rows:
            assert r["dataset"] == dataset, f"mixed datasets in {src}"
            for key, seen in ((r["id"], seen_ids), (r["problem"].strip(), seen_problems)):
                assert key not in seen, f"{r['id']} in {name} overlaps with {seen.get(key)}"
                seen[key] = name
            r["split"] = split
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name:16s} {len(rows):5d}  <- {src}")

    print(f"total {len(seen_ids)} questions, no overlap across files")


if __name__ == "__main__":
    main()
