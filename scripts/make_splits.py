"""
Build train/dev/test splits for GSM8K, MATH, and AIME 2024.

Usage:
    pip install datasets
    python data/make_splits.py --out data --seed 0
    rm -f data/.DS_Store
    COPYFILE_DISABLE=1 tar -czf data.tar.gz data

Optional: --max-train N subsamples each train split to N questions (to cap trace-generation compute).

Output: data/<dataset>/{train,dev,test}.jsonl plus data/stats.json
Each line: {"id", "dataset", "problem", "answer", "solution", "subject", "level"}
"""
import argparse
import json
import os
import random

from datasets import load_dataset, concatenate_datasets

MATH_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
]


def last_boxed(text):
    """Return contents of the last \\boxed{...} in text, handling nested braces."""
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


def all_boxed(text):
    """Return contents of every \\boxed{...} in text."""
    out, pos = [], 0
    while True:
        start = text.find("\\boxed", pos)
        if start == -1:
            return out
        i = text.find("{", start)
        if i == -1:
            return out
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[i + 1:j].strip())
                    break
        pos = j + 1


def clean_answer(ans):
    """Strip trailing punctuation that is not part of the answer."""
    return ans.strip().rstrip(".,").strip()


def parse_level(level):
    """'Level 5' or 5 -> 5; 'Level ?' or missing -> None."""
    if level is None:
        return None
    digits = "".join(ch for ch in str(level) if ch.isdigit())
    return int(digits) if digits else None


def get(row, *keys):
    """Fetch the first matching key, tolerant of capitalization differences."""
    for k in keys:
        for cand in (k, k.capitalize(), k.upper()):
            if cand in row and row[cand] is not None:
                return row[cand]
    return None


def carve_dev(items, frac, rng):
    items = items[:]
    rng.shuffle(items)
    n_dev = round(len(items) * frac)
    return items[n_dev:], items[:n_dev]


def build_gsm8k():
    ds = load_dataset("openai/gsm8k", "main")
    def conv(split):
        out = []
        for i, r in enumerate(ds[split]):
            sol, ans = r["answer"].split("####")
            out.append({"id": f"gsm8k-{split}-{i}", "dataset": "gsm8k",
                        "problem": r["question"], "answer": ans.strip().replace(",", ""),
                        "solution": sol.strip(), "subject": None, "level": None})
        return out
    return conv("train"), conv("test")


def build_math():
    train_raw = concatenate_datasets(
        [load_dataset("EleutherAI/hendrycks_math", s, split="train") for s in MATH_SUBJECTS])
    train = []
    dropped = {"no_answer": 0, "multi_part_answer": 0}
    for i, r in enumerate(train_raw):
        boxed = [b for b in all_boxed(r["solution"]) if b]
        if not boxed:
            dropped["no_answer"] += 1
            continue
        if len(set(boxed)) > 1:  # several distinct boxed parts; last one alone is incomplete
            dropped["multi_part_answer"] += 1
            continue
        train.append({"id": f"math-train-{i}", "dataset": "math", "problem": r["problem"],
                      "answer": clean_answer(boxed[-1]), "solution": r["solution"],
                      "subject": r.get("type"), "level": parse_level(r.get("level"))})
    test = []
    for r in load_dataset("HuggingFaceH4/MATH-500", split="test"):
        test.append({"id": r["unique_id"], "dataset": "math500", "problem": r["problem"],
                     "answer": clean_answer(r["answer"]), "solution": r["solution"],
                     "subject": r["subject"], "level": parse_level(r["level"])})
    return train, test, dropped


def build_aime():
    test = []
    for i, r in enumerate(load_dataset("HuggingFaceH4/aime_2024", split="train")):
        test.append({"id": f"aime2024-{get(r, 'id') or i}", "dataset": "aime2024",
                     "problem": get(r, "problem"), "answer": str(int(str(get(r, "answer")).strip())),
                     "solution": get(r, "solution"), "subject": None, "level": None})
    return test


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def check_no_overlap(splits, name):
    seen = {}
    for split, rows in splits.items():
        for r in rows:
            key = " ".join(r["problem"].split()).lower()
            if key in seen and seen[key] != split:
                raise ValueError(f"{name}: problem appears in both {seen[key]} and {split}")
            seen[key] = split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dev-frac", type=float, default=0.1)
    p.add_argument("--max-train", type=int, default=None)
    args = p.parse_args()
    rng = random.Random(args.seed)
    stats = {}

    gsm_trainval, gsm_test = build_gsm8k()
    math_trainval, math_test, math_dropped = build_math()
    aime_test = build_aime()

    for name, trainval, test in [("gsm8k", gsm_trainval, gsm_test),
                                 ("math", math_trainval, math_test)]:
        train, dev = carve_dev(trainval, args.dev_frac, rng)
        if args.max_train:
            train = train[:args.max_train]
        splits = {"train": train, "dev": dev, "test": test}
        check_no_overlap(splits, name)
        for split, rows in splits.items():
            write(os.path.join(args.out, name, f"{split}.jsonl"), rows)
        stats[name] = {k: len(v) for k, v in splits.items()}

    write(os.path.join(args.out, "aime", "test.jsonl"), aime_test)
    stats["aime"] = {"train": 0, "dev": 0, "test": len(aime_test)}
    stats["math"]["dropped"] = math_dropped

    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
