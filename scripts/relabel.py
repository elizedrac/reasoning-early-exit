"""
Re-ground, re-merge and re-judge already labeled traces without calling the LLM.

Takes label_chunks.py output, rewrites Unicode answers as LaTeX, drops piece answers that the
piece doesn't state itself, merges pieces into chunks again, and judges them with math-verify
(steps 3 to 6 in label_chunks.py). It uses piece_answers_raw if present, else piece_answers, so
it works on files labeled before grounding existed and is safe to rerun on its own output.
The file is rewritten in place (via a temp file). The raw extractor answers stay in
piece_answers_raw, so nothing is lost. It refuses a file modified in the last 10 minutes, which
may still be being labeled (for example on Colab); --force overrides.

Chunk boundaries change, so hidden states saved from the old chunks must be redone: delete that
file's hidden/<name>/ folder and rerun extract_hidden.py (it skips traces already in a shard).

Usage:
    python scripts/relabel.py labeled/gsm8k_train.jsonl [labeled/math_train.jsonl ...]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_traces import MODEL as GEN_MODEL  # noqa: E402
from label_chunks import build_record, summarize  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="label_chunks.py output files, fixed in place")
    p.add_argument("--force", action="store_true", help="also fix files modified in the last 10 minutes")
    args = p.parse_args()
    from transformers import AutoTokenizer
    gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    for path in args.files:
        age = time.time() - os.path.getmtime(path)
        if age < 600 and not args.force:
            print(f"SKIP {path}: modified {age / 60:.0f} min ago, may still be being labeled (--force to fix anyway)")
            continue
        relabel(path, gen_tok)


def relabel(path, gen_tok):
    mismatch_path = path + ".mismatches.jsonl"
    if os.path.exists(mismatch_path):
        os.remove(mismatch_path)

    def log(entry):
        with open(mismatch_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    n = n_bad = n_before = n_after = 0
    t0 = time.time()
    with open(path + ".tmp", "w") as f:
        for line in open(path):
            if not line.strip().endswith("}"):
                continue  # a partial last line from an interrupted run
            rec = json.loads(line)
            raw = rec.get("piece_answers_raw", rec["piece_answers"])
            sources = rec.get("piece_sources", ["llm"] * len(raw))
            labeler = rec.get("labeler")
            for key in ("piece_answers_raw", "piece_answers", "piece_sources", "labeler"):
                rec.pop(key, None)
            n_before += len(rec["chunks"])
            out, bad = build_record(rec, raw, sources, labeler, gen_tok, log)
            n_after += len(out["chunks"])
            n_bad += bad
            n += 1
            f.write(json.dumps(out) + "\n")
    os.replace(path + ".tmp", path)
    print(f"{n} traces in {time.time() - t0:.0f} s; chunks {n_before} -> {n_after} "
          f"({n_before / max(n, 1):.1f} -> {n_after / max(n, 1):.1f} per trace)")
    if n_bad:
        print(f"WARNING: {n_bad} chunk token spans did not match their text; see {mismatch_path}")
    summarize(path)


if __name__ == "__main__":
    main()
