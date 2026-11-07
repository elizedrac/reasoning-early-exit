"""
Strong baseline: a correctness probe on hidden states, used for early exit
(Zhang et al. 2025, "Reasoning Models Know When They're Right").

Steps:
    1. Load hidden-state shards from scripts/extract_hidden.py (X [n, num_layers, 1536], labels).
    2. Train a probe (logistic regression or 2-layer MLP) on train, early-stop on dev.
       Loss: weighted BCE, positive-class weight w = alpha * (#neg / #pos).
       Optimizer: Adam.
    3. Simulate early exit on test traces: stop at the first chunk whose probe score >= tau,
       answer with that chunk's answer, tokens_used = that chunk's end_token. Sweep tau.

Usage:
    python baselines/strong-baseline.py --train hidden/train --dev hidden/dev --test hidden/test \
        --traces traces/gsm8k_test.jsonl --probe mlp --out results/probe_gsm8k_test.jsonl
"""
import argparse


def load_hidden(shard_dir, layer=-1):
    """TODO: concatenate X and meta from all shard_*.pt in shard_dir; select one layer
    (index into the shard's `layers` list). Return X [n, 1536] float32, y [n], meta."""
    raise NotImplementedError


class LinearProbe:
    """TODO: logistic regression, nn.Linear(1536, 1)."""

    def __init__(self, dim):
        raise NotImplementedError


class MLPProbe:
    """TODO: 2-layer MLP, Linear(dim, hidden) -> ReLU -> Linear(hidden, 1)."""

    def __init__(self, dim, hidden=256):
        raise NotImplementedError


def pos_weight(y, alpha=1.0):
    """TODO: alpha * (#negatives / #positives), passed to BCEWithLogitsLoss(pos_weight=...)."""
    raise NotImplementedError


def train_probe(probe, X_train, y_train, X_dev, y_dev, alpha=1.0, lr=1e-3, epochs=100, patience=5):
    """TODO: Adam + weighted BCE; evaluate dev loss each epoch; stop after `patience` epochs
    without improvement and restore the best weights. Return the trained probe and history."""
    raise NotImplementedError


def predict(probe, X):
    """TODO: sigmoid(probe(X)) as probabilities."""
    raise NotImplementedError


def simulate_early_exit(traces, scores_by_chunk, tau):
    """TODO: for each trace, stop at the first labeled chunk with score >= tau
    (answer = chunk answer, tokens_used = chunk end_token); if none, use full reasoning.
    Return predictions in the eval/score.py format."""
    raise NotImplementedError


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--traces", required=True, help="test traces JSONL for simulating early exit")
    p.add_argument("--probe", choices=["linear", "mlp"], default="linear")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--tau", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    # TODO: load, train, score test chunks, simulate early exit for each tau, write predictions.
    raise NotImplementedError


if __name__ == "__main__":
    main()
