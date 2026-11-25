"""
Extension: a correctness probe on hidden states every N tokens, used for early exit mid-chunk.
SKELETON ONLY: function bodies are not written yet.

Same probe as baselines/strong-baseline.py, but trained on the positions saved by
`scripts/extract_hidden.py --every 8` (every 8 generated tokens inside each labeled chunk, plus each
chunk's end; each position has the label of its chunk). The difference is the stopping rule:

    stop at the first position whose probe score >= tau, or, with --consecutive m, the first
    position where the last m positions in a row all scored >= tau.

If the stop is a chunk end, that chunk's extracted answer is used (as in the strong baseline).
If the stop is mid-chunk there is no extracted answer, so the model is forced to answer at that
token (force_answers from baselines/simple-baseline.py, needs MLX on Apple silicon or vLLM on a GPU).
Without --force, mid-chunk stops use the latest answer stated by that token (as the budget rule
does without --force), which needs no model.

Steps:
    1. Load <data>/hidden_tokens/<name>/ for train, dev, test; standardize with train statistics.
    2. Train a linear or MLP probe (optionally grid-searched with --tune, lowest dev Brier).
    3. For each tau: find each trace's stop position (pass 1, no model).
    4. With --force: force answers at every mid-chunk stop not already in the cache (pass 2).
    5. Pick tau on dev (largest token reduction with dev accuracy within --max-drop of full),
       write predictions per tau for eval/score.py and summary.json.

Usage:
    python extensions/token-probe.py --data <cis530 folder> --probe linear --out-dir results/token_probe
    python extensions/token-probe.py --data <cis530 folder> --probe linear --force --backend mlx \
        --out-dir results/token_probe_forced
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from score import gold_answers, probe_metrics, score  # noqa: E402


def load_module(name, path):
    """The baseline files have hyphens in their names, so they can't be imported normally."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reused as-is: linear_probe, mlp_probe, train_probe, predict, load_labeled, GRID, TAUS.
strong = load_module("strong_baseline", "baselines/strong-baseline.py")
# Reused as-is: load_backend, force_answers.
simple = load_module("simple_baseline", "baselines/simple-baseline.py")


def load_hidden_tokens(shard_dir, layer=-1):
    """Like strong.load_hidden, but keys are (trace id, chunk k, token) using meta["token"], and
    also returns whether each row is its chunk's end (token == end_token, or the answered end for
    the last chunk).
    NOTE: strong.load_hidden can't be reused directly: it keys rows by (id, k), which is not unique
    here. Alternative: add a `key_fields` argument to strong.load_hidden and call that."""
    raise NotImplementedError


def load_split(data, names, mean=None, std=None):
    """Same as strong.load_split, but reads hidden_tokens/ with load_hidden_tokens.
    NOTE: strong.load_split hardcodes the hidden/ folder and load_hidden. Alternative: give it
    `folder` and `loader` arguments and reuse it here."""
    raise NotImplementedError


def stop_positions(traces, scores, tau, consecutive=1):
    """Pass 1 (no model). scores: {(id, k, token): probe probability}. For each trace, walk its
    positions in token order and stop at the first one where the last `consecutive` scores are all
    >= tau. Returns {id: (token, chunk, is_chunk_end)} or {id: None} if nothing fires."""
    raise NotImplementedError


def answers_at_stops(traces, stops, forced=None):
    """Answer and tokens_used for each trace:
    - no stop: full reasoning (strong.simulate_early_exit's fallback)
    - stop at a chunk end: that chunk's answer
    - stop mid-chunk: forced[(id, token)] with --force, else the latest answer stated by that token
      (as simple.fixed_budget does without --force; None if none, scored as wrong).
    Output rows have the fields eval/score.py needs, with probe_scores / chunk_labels over positions."""
    raise NotImplementedError


def force_at_stops(backend, traces, stops, cache_path):
    """Pass 2 (needs the model). Force an answer at every mid-chunk stop.
    Reuses simple.force_answers by passing the stop token as its budget N.
    NOTE: force_answers takes one budget for all traces it is given, but stops differ per trace.
    Either group traces by stop token and call it once per group (many small batches), or change
    force_answers to accept {id: N} (small change, simple-baseline.py keeps passing an int).
    Answers are cached in cache_path as {"<id>:<token>": answer}, since many taus share stops and
    reruns should not regenerate. Returns {(id, token): answer}."""
    raise NotImplementedError


def pick_tau(dev_results, full_acc, max_drop):
    """Largest dev token reduction with accuracy >= full_acc - max_drop (as in strong-baseline.py).
    NOTE: strong-baseline.py does this inline in main(); could be pulled out there and imported."""
    raise NotImplementedError


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="folder with hidden_tokens/ and labeled/ (the shared cis530 folder)")
    p.add_argument("--train", nargs="+", default=["gsm8k_train", "math_train"])
    p.add_argument("--dev", nargs="+", default=["gsm8k_dev", "math_dev"])
    p.add_argument("--test", nargs="+", default=["gsm8k_test", "math500_test", "aime2024_test"])
    p.add_argument("--probe", choices=["linear", "mlp"], default="linear")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256, help="MLP hidden size")
    p.add_argument("--tune", action="store_true", help="grid-search the probe on dev (strong.GRID)")
    p.add_argument("--tau", type=float, nargs="+", default=strong.TAUS)
    p.add_argument("--consecutive", type=int, default=1, help="stop only after m positions in a row >= tau")
    p.add_argument("--max-drop", type=float, default=0.01, help="max dev accuracy drop when picking tau")
    p.add_argument("--force", action="store_true", help="force answers at mid-chunk stops (needs the model)")
    p.add_argument("--backend", choices=["vllm", "mlx"], default="mlx", help="for --force")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    # 1. load_split(train), then dev and test with the train mean/std.
    # 2. strong.train_probe (or grid over strong.GRID by dev Brier), strong.predict on dev and test.
    # 3. For each split and tau: stop_positions.
    # 4. If --force: simple.load_backend sized for the largest stop token, then force_at_stops.
    #    Forcing on dev is needed for every tau, because tau is picked on dev accuracy.
    # 5. answers_at_stops -> score -> pick_tau on dev; write predictions and summary.json.
    raise NotImplementedError


if __name__ == "__main__":
    main()
