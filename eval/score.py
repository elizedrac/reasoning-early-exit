"""
Score early-exit predictions against gold answers.

Inputs:
    --pred   JSONL, one line per question, written by baselines/:
             {"id", "answer", "tokens_used", "full_tokens",
              optional "probe_scores" and "chunk_labels" (lists over the trace's labeled chunks)}
    --gold   JSONL with "id" and the gold answer under "gold" or "answer"
             (labeled/*.jsonl, traces/*.jsonl or data/trace_inputs/*.jsonl all work)

Metrics (see eval/scoring.md):
    accuracy            fraction of questions whose answer math-verify judges equal to gold
                        (a missing or unparseable answer counts as wrong)
    avg_tokens          mean generated tokens used before stopping
    token_reduction     1 - sum(tokens_used) / sum(full_tokens), vs. full reasoning
    probe metrics       ROC-AUC, ECE, Brier score of chunk-level probe scores vs. chunk labels
                        (only if the predictions carry probe_scores)

Usage:
    python eval/score.py --pred results/first_gsm8k_test.jsonl --gold data/trace_inputs/gsm8k_test.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from generate_traces import is_correct  # noqa: E402  (same judge used for the labels)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def gold_answers(rows):
    return {r["id"]: str(r["gold"] if "gold" in r else r["answer"]) for r in rows}


def accuracy(preds, golds):
    """Fraction of predictions judged equal to gold by math-verify. Missing or unparseable
    predictions count as wrong."""
    # math-verify returns True only for an equivalent answer, so None or a parse failure is wrong.
    hits = [p["answer"] is not None and is_correct(p["answer"], golds[p["id"]]) is True for p in preds]
    return float(np.mean(hits))


def avg_tokens(preds):
    return float(np.mean([p["tokens_used"] for p in preds]))


def token_reduction(preds):
    return 1 - sum(p["tokens_used"] for p in preds) / sum(p["full_tokens"] for p in preds)


def roc_auc(scores, labels):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def expected_calibration_error(scores, labels, n_bins=10):
    """Equal-width bins: sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, float)
    # Bin index floor(score * 10); a score of exactly 1.0 goes in the top bin. Empty bins are skipped.
    bins = np.minimum((scores * n_bins).astype(int), n_bins - 1)
    return float(sum((bins == b).mean() * abs(labels[bins == b].mean() - scores[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))


def brier_score(scores, labels):
    return float(np.mean((np.asarray(scores, float) - np.asarray(labels, float)) ** 2))


def probe_metrics(scores, labels):
    return {"roc_auc": roc_auc(scores, labels), "ece": expected_calibration_error(scores, labels),
            "brier": brier_score(scores, labels)}


def score(preds, golds):
    missing = [p["id"] for p in preds if p["id"] not in golds]
    assert not missing, f"{len(missing)} predictions have no gold answer, e.g. {missing[:3]}"
    m = {"n": len(preds), "accuracy": accuracy(preds, golds), "avg_tokens": avg_tokens(preds),
         "token_reduction": token_reduction(preds)}
    scores = [s for p in preds for s in p.get("probe_scores", [])]
    labels = [y for p in preds for y in p.get("chunk_labels", [])]
    # Probe metrics only for probe predictions, and AUC needs both correct and incorrect chunks.
    if scores and len(set(labels)) == 2:
        m.update(probe_metrics(scores, labels))
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--out", default=None, help="optional JSON file for the metrics")
    args = p.parse_args()
    m = score(load_jsonl(args.pred), gold_answers(load_jsonl(args.gold)))
    for k, v in m.items():
        print(f"{k:16s} {v:.4f}" if isinstance(v, float) else f"{k:16s} {v}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(m, f, indent=2)


if __name__ == "__main__":
    main()
