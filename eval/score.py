"""
Score early-exit predictions against gold answers.

Inputs:
    --pred   JSONL, one line per (question, setting):
                 {"id", "answer", "tokens_used",
                  optional "setting"       e.g. "full", "first", "budget=2048", "tau=0.9";
                                           rows are grouped by it and scored separately
                                           (missing -> "default"),
                  optional "probe_scores"  list of floats, one per chunk (probe P(correct)),
                  optional "chunk_labels"  list of 1 / 0 / null, same length as probe_scores}
             "answer" is the bare answer string (no \\boxed{}), or null if there is none.
    --gold   JSONL with {"id", "answer"} (data/trace_inputs/*.jsonl) or {"id", "gold"}
             (traces from generate_traces.py)
    --full   optional traces JSONL (generate_traces.py output): full-reasoning token counts for
             token reduction, and full-reasoning accuracy on the same questions for reference

Metrics (see eval/scoring.md):
    accuracy            fraction of gold questions whose answer math-verify judges equal to gold
                        (missing or unparseable predictions count as wrong)
    avg_tokens          mean tokens_used over predicted questions
    token_reduction     1 - sum(tokens_used) / sum(full num_tokens), over shared IDs
    probe metrics       ROC-AUC, ECE, Brier score of chunk-level probe scores vs. chunk labels
                        (pooled over all chunks with a non-null label, each question counted once)

Usage:
    python eval/score.py --pred results/x.jsonl --gold data/trace_inputs/gsm8k_test.jsonl \
        --full traces/gsm8k_test.jsonl --out results/x_metrics.json
"""
import argparse
import json
import os
import sys
from functools import lru_cache

import numpy as np

# Judge answers exactly as the labeling pipeline does (scripts/generate_traces.py).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from generate_traces import is_correct  # noqa: E402


def load_jsonl(path):
    """Read a JSONL file into a list of dicts (blank lines skipped)."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@lru_cache(maxsize=None)
def judge(pred, gold):
    """True only if math-verify says pred == gold; None/unparseable -> False."""
    return pred is not None and is_correct(str(pred), str(gold)) is True


def gold_answers(golds):
    """{id: gold answer string} from rows with "answer" (inputs) or "gold" (traces)."""
    out = {}
    for g in golds:
        ans = g["answer"] if "answer" in g else g["gold"]
        out[g["id"]] = str(ans)
    return out


def accuracy(preds, golds):
    """Fraction of gold IDs whose prediction math-verify judges correct.
    Missing or unparseable predictions count as wrong."""
    pred_by_id = {p["id"]: p.get("answer") for p in preds}
    if not golds:
        return None
    return sum(judge(pred_by_id.get(i), g) for i, g in golds.items()) / len(golds)


def avg_tokens(preds):
    """Mean of tokens_used over questions."""
    return float(np.mean([p["tokens_used"] for p in preds])) if preds else None


def token_reduction(preds, full_traces):
    """1 - sum(tokens_used) / sum(num_tokens of the full trace), over shared IDs."""
    full = {t["id"]: t["num_tokens"] for t in full_traces}
    shared = [p for p in preds if p["id"] in full]
    denom = sum(full[p["id"]] for p in shared)
    if not denom:
        return None
    return 1 - sum(p["tokens_used"] for p in shared) / denom


def full_answer(trace):
    """The full-reasoning answer: final answer, or the forced answer if the trace was cut off."""
    return trace["forced_answer"] if trace.get("truncated") else trace.get("final_answer")


def roc_auc(scores, labels):
    """Area under the ROC curve; None if only one class is present."""
    from sklearn.metrics import roc_auc_score
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def expected_calibration_error(scores, labels, n_bins=10):
    """ECE with equal-width bins: sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|.
    Bin b holds scores in [b/n_bins, (b+1)/n_bins); a score of exactly 1.0 goes in the last bin."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
    if len(s) == 0:
        return None
    bins = np.minimum((s * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - s[mask].mean())
    return float(ece)


def brier_score(scores, labels):
    """Mean squared error between probe probability and the 0/1 label."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
    return float(np.mean((s - y) ** 2)) if len(s) else None


def chunk_pairs(preds):
    """(scores, labels) over chunks with a non-null label, each question counted once even if it
    appears under several settings."""
    scores, labels, seen = [], [], set()
    for p in preds:
        if p["id"] in seen or p.get("probe_scores") is None or p.get("chunk_labels") is None:
            continue
        seen.add(p["id"])
        if len(p["probe_scores"]) != len(p["chunk_labels"]):
            raise ValueError(f"{p['id']}: {len(p['probe_scores'])} probe_scores vs "
                             f"{len(p['chunk_labels'])} chunk_labels")
        for s, y in zip(p["probe_scores"], p["chunk_labels"]):
            if y is not None:
                scores.append(float(s))
                labels.append(int(y))
    return scores, labels


def score_setting(preds, golds, full_traces):
    unknown = [p["id"] for p in preds if p["id"] not in golds]
    if unknown:
        print(f"warning: {len(unknown)} predicted IDs not in gold (e.g. {unknown[0]}), ignored",
              file=sys.stderr)
    preds = [p for p in preds if p["id"] in golds]
    m = {"n_gold": len(golds), "n_pred": len(preds), "accuracy": accuracy(preds, golds),
         "avg_tokens": avg_tokens(preds)}
    if full_traces is not None:
        m["token_reduction"] = token_reduction(preds, full_traces)
    return m


def full_reference(golds, full_traces):
    """Accuracy and avg tokens of full reasoning on the gold questions it covers."""
    ts = [t for t in full_traces if t["id"] in golds]
    if not ts:
        return None
    return {"n": len(ts), "accuracy": sum(judge(full_answer(t), golds[t["id"]]) for t in ts) / len(ts),
            "avg_tokens": float(np.mean([t["num_tokens"] for t in ts]))}


def fmt(x, pct=False):
    if x is None:
        return "-"
    if isinstance(x, int):
        return str(x)
    return f"{x:.2%}" if pct else f"{x:.4f}" if abs(x) < 10 else f"{x:.0f}"


def print_table(results):
    cols = [("setting", None), ("n_pred", False), ("accuracy", True), ("avg_tokens", False),
            ("token_reduction", True)]
    cols = [c for c in cols if c[0] == "setting" or any(c[0] in r for r in results["settings"].values())]
    rows = [[name] + [fmt(m.get(c), pct) for c, pct in cols[1:]] for name, m in results["settings"].items()]
    if results.get("full_reference"):
        f = results["full_reference"]
        rows.append(["(full traces)", fmt(f["n"]), fmt(f["accuracy"], True), fmt(f["avg_tokens"])]
                    + ["-"] * (len(cols) - 4))
    header = [c for c, _ in cols]
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for r in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))
    if results.get("probe"):
        p = results["probe"]
        print(f"\nprobe ({p['n_chunks']} chunks, {p['pos_rate']:.1%} positive): "
              f"ROC-AUC {fmt(p['roc_auc'])}  ECE {fmt(p['ece'])}  Brier {fmt(p['brier'])}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--full", default=None)
    p.add_argument("--bins", type=int, default=10, help="ECE bins (equal width)")
    p.add_argument("--out", default=None, help="optional JSON file for the metrics")
    args = p.parse_args()

    preds = load_jsonl(args.pred)
    golds = gold_answers(load_jsonl(args.gold))
    full_traces = load_jsonl(args.full) if args.full else None

    by_setting = {}
    for row in preds:
        by_setting.setdefault(str(row.get("setting", "default")), []).append(row)
    results = {"pred": args.pred, "gold": args.gold, "full": args.full,
               "settings": {s: score_setting(rows, golds, full_traces) for s, rows in by_setting.items()}}
    for s, m in results["settings"].items():
        if m["n_pred"] < m["n_gold"]:
            print(f"warning: setting {s!r} predicts {m['n_pred']} of {m['n_gold']} gold questions; "
                  f"missing ones count as wrong", file=sys.stderr)
    if full_traces is not None:
        results["full_reference"] = full_reference(golds, full_traces)

    scores, labels = chunk_pairs([r for r in preds if r["id"] in golds])
    if scores:
        results["probe"] = {"n_chunks": len(scores), "pos_rate": float(np.mean(labels)),
                            "roc_auc": roc_auc(scores, labels),
                            "ece": expected_calibration_error(scores, labels, args.bins),
                            "brier": brier_score(scores, labels)}

    print_table(results)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
