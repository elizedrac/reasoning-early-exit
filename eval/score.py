"""
Score early-exit predictions against gold answers.

Inputs:
    --pred   JSONL, one line per question: {"id", "answer", "tokens_used", optional "probe_scores",
             optional "chunk_labels"}  (format TODO: finalize with baselines/)
    --gold   JSONL with {"id", "answer"} (e.g. data/trace_inputs/*.jsonl)
    --full   optional traces JSONL (generate_traces.py output) giving full-reasoning token counts,
             used for % token reduction

Metrics (see eval/scoring.md):
    accuracy            fraction of questions whose answer math-verify judges equal to gold
    avg_tokens          mean reasoning tokens used before stopping
    token_reduction     1 - sum(tokens_used) / sum(full tokens), vs. full reasoning
    probe metrics       ROC-AUC, ECE, Brier score of chunk-level probe scores vs. chunk labels

Usage:
    python eval/score.py --pred results/x.jsonl --gold data/trace_inputs/gsm8k_test.jsonl \
        --full traces/gsm8k_test.jsonl
"""
import argparse


def load_jsonl(path):
    """TODO: read a JSONL file into a list of dicts."""
    raise NotImplementedError


def accuracy(preds, golds):
    """TODO: fraction of IDs where math_verify.verify(parse(gold), parse(pred)) is True.
    Missing or unparseable predictions count as wrong."""
    raise NotImplementedError


def avg_tokens(preds):
    """TODO: mean of tokens_used over questions."""
    raise NotImplementedError


def token_reduction(preds, full_traces):
    """TODO: 1 - sum(tokens_used) / sum(num_tokens of the full trace), over shared IDs."""
    raise NotImplementedError


def roc_auc(scores, labels):
    """TODO: sklearn.metrics.roc_auc_score on chunk-level probe scores."""
    raise NotImplementedError


def expected_calibration_error(scores, labels, n_bins=10):
    """TODO: ECE with equal-width bins: sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|."""
    raise NotImplementedError


def brier_score(scores, labels):
    """TODO: mean((score - label)^2)."""
    raise NotImplementedError


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--full", default=None)
    p.add_argument("--out", default=None, help="optional JSON file for the metrics")
    args = p.parse_args()
    # TODO: load files, compute each metric, print a table, optionally write --out.
    raise NotImplementedError


if __name__ == "__main__":
    main()
