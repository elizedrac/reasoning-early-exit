"""
Strong baseline: a correctness probe on hidden states, used for early exit
(Zhang et al. 2025, "Reasoning Models Know When They're Right").

Steps:
    1. Load hidden-state shards from scripts/extract_hidden.py (one vector per labeled chunk, last
       layer) and standardize them with train statistics.
    2. Train a probe (logistic regression or 2-layer MLP) on the train sets, early-stopping on dev
       loss. Loss: weighted BCE, positive-class weight w = alpha * (#neg / #pos). Optimizer: Adam.
       With --tune, lr, weight decay, alpha (and the MLP hidden size) are grid-searched (GRID) and
       the config with the lowest dev Brier score is kept (Brier rewards both ranking and
       calibration; picking by ROC-AUC alone chose configs with poor calibration).
    3. Simulate early exit: stop at the first labeled chunk whose probe score >= tau, answer with
       that chunk's answer, tokens_used = its end_token; if no chunk fires, use full reasoning.
       tau is swept, and one tau is picked on dev: the one with the largest token reduction whose
       dev accuracy is within --max-drop of full reasoning.

Reads <data>/hidden/<name>/ and <data>/labeled/<name>.jsonl for every name given.
Writes <out-dir>/probe_<name>_tau<tau>.jsonl (predictions for eval/score.py) and
<out-dir>/summary.json.

Usage:
    python baselines/strong-baseline.py --data <cis530 dir> --probe linear --out-dir results/probe
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import torch
from torch import nn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "baselines"))
from score import gold_answers, probe_metrics, score  # noqa: E402

TAUS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
GRID = {"lr": [1e-4, 1e-3, 1e-2], "weight_decay": [0, 1e-4, 1e-3], "alpha": [0.5, 1.0, 2.0],
        "hidden": [128, 256, 512]}


def load_hidden(shard_dir, layer=-1):
    """Concatenate all shards in shard_dir and select one layer (index into the shard's `layers`).
    Returns X [n, 1536] float32, y [n] float32, and keys [(trace id, chunk k)]."""
    X, y, keys = [], [], []
    for path in sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt"))):
        d = torch.load(path, weights_only=False)
        if len(d["X"]) == 0:
            continue
        X.append(d["X"][:, layer].float())
        y += d["meta"]["label"]
        keys += list(zip(d["meta"]["id"], d["meta"]["k"]))
    assert X, f"no hidden states in {shard_dir}"
    return torch.cat(X), torch.tensor(y, dtype=torch.float32), keys


def load_labeled(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def linear_probe(dim):
    return nn.Linear(dim, 1)


def mlp_probe(dim, hidden=256):
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def pos_weight(y, alpha=1.0):
    # Positive-class weight alpha * #neg / #pos: alpha = 1 balances the classes, > 1 favors positives.
    return torch.tensor(alpha * float((y == 0).sum()) / float((y == 1).sum()))


def train_probe(probe, X_train, y_train, X_dev, y_dev, alpha=1.0, lr=1e-3, weight_decay=1e-4,
                epochs=200, batch_size=512, patience=10, seed=0):
    """Adam + weighted BCE. Dev loss (same weighting) is checked every epoch; training stops after
    `patience` epochs without improvement and the best weights are restored."""
    torch.manual_seed(seed)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight(y_train, alpha))
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        probe.train()
        for idx in torch.randperm(len(X_train)).split(batch_size):
            opt.zero_grad()
            loss_fn(probe(X_train[idx]).squeeze(1), y_train[idx]).backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            dev_loss = loss_fn(probe(X_dev).squeeze(1), y_dev).item()
        history.append(dev_loss)
        # Early stopping: keep the weights with the lowest dev loss, stop after `patience` bad epochs.
        if dev_loss < best - 1e-5:
            best, best_state, bad = dev_loss, {k: v.clone() for k, v in probe.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    probe.load_state_dict(best_state)
    return probe, history


def predict(probe, X):
    probe.eval()
    with torch.no_grad():
        return torch.sigmoid(probe(X).squeeze(1)).numpy()


def simulate_early_exit(traces, scores, tau):
    """scores: {(trace id, k): probe probability} for labeled chunks. Stop at the first such chunk
    with score >= tau; if none fires, use full reasoning."""
    preds = []
    for t in traces:
        chunk_scores, chunk_labels, stop = [], [], None
        for c in t["chunks"]:
            s = scores.get((t["id"], c["k"]))
            if s is None:  # chunk with no answer, so no hidden state or score
                continue
            chunk_scores.append(float(s))
            chunk_labels.append(int(c["correct"]))
            # Early exit: the first chunk whose score reaches tau gives the answer and token count.
            if stop is None and s >= tau:
                stop = c
        if stop is not None:
            answer, used = stop["answer"], stop["end_token"]
        else:
            answer = t["forced_answer"] if t["truncated"] else t["final_answer"]
            used = t["num_tokens"]
        preds.append({"id": t["id"], "dataset": t["dataset"], "rule": f"probe_tau{tau}", "answer": answer,
                      "tokens_used": used, "full_tokens": t["num_tokens"],
                      "probe_scores": chunk_scores, "chunk_labels": chunk_labels})
    return preds


def full_accuracy(traces, golds):
    # tau = 2 is never reached, so every trace uses full reasoning.
    return score(simulate_early_exit(traces, {}, 2.0), golds)["accuracy"]


def load_split(data, names, mean=None, std=None):
    X, y, keys = zip(*(load_hidden(os.path.join(data, "hidden", n)) for n in names))
    X, y, keys = torch.cat(X), torch.cat(y), sum(keys, [])
    # Standardize each feature with train mean and std (computed once on train, reused for dev and test).
    if mean is None:
        mean, std = X.mean(0), X.std(0) + 1e-6
    return (X - mean) / std, y, keys, mean, std


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="folder with hidden/ and labeled/ (the shared cis530 folder)")
    p.add_argument("--train", nargs="+", default=["gsm8k_train", "math_train"])
    p.add_argument("--dev", nargs="+", default=["gsm8k_dev", "math_dev"])
    p.add_argument("--test", nargs="+", default=["gsm8k_test", "math500_test", "aime2024_test"])
    p.add_argument("--probe", choices=["linear", "mlp"], default="linear")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256, help="MLP hidden size")
    p.add_argument("--tune", action="store_true",
                   help="grid-search lr, weight decay, alpha (and MLP hidden size) by dev Brier score")
    p.add_argument("--tau", type=float, nargs="+", default=TAUS)
    p.add_argument("--max-drop", type=float, default=0.01, help="max dev accuracy drop when picking tau")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    X_tr, y_tr, _, mean, std = load_split(args.data, args.train)
    X_dev, y_dev, keys_dev, _, _ = load_split(args.data, args.dev, mean, std)
    print(f"train {len(X_tr)} chunks ({y_tr.mean():.1%} correct), dev {len(X_dev)} ({y_dev.mean():.1%})")
    def fit(cfg):
        probe = linear_probe(X_tr.shape[1]) if args.probe == "linear" else mlp_probe(X_tr.shape[1], cfg["hidden"])
        probe, history = train_probe(probe, X_tr, y_tr, X_dev, y_dev, alpha=cfg["alpha"], lr=cfg["lr"],
                                     weight_decay=cfg["weight_decay"])
        return probe, history, probe_metrics(predict(probe, X_dev), y_dev.numpy())

    cfg = {"lr": args.lr, "weight_decay": args.weight_decay, "alpha": args.alpha, "hidden": args.hidden}
    grid_results = []
    if args.tune:
        hiddens = GRID["hidden"] if args.probe == "mlp" else [args.hidden]
        grid = [dict(lr=lr, weight_decay=wd, alpha=a, hidden=h) for lr, wd, a, h in
                itertools.product(GRID["lr"], GRID["weight_decay"], GRID["alpha"], hiddens)]
        best_brier = float("inf")
        for i, c in enumerate(grid):
            probe_c, history_c, m = fit(c)
            grid_results.append({**c, "epochs": len(history_c), **{f"dev_{k}": v for k, v in m.items()}})
            print(f"  [{i + 1}/{len(grid)}] {c} epochs {len(history_c)} dev AUC {m['roc_auc']:.4f} "
                  f"ECE {m['ece']:.4f} Brier {m['brier']:.4f}", flush=True)
            # Keep the config with the lowest dev Brier score (rewards both ranking and calibration).
            if m["brier"] < best_brier:
                best_brier, cfg, probe, history = m["brier"], c, probe_c, history_c
        print(f"best config {cfg} (dev Brier {best_brier:.4f})")
    else:
        probe, history, _ = fit(cfg)
    print(f"trained {len(history)} epochs, best dev loss {min(history):.4f}")

    # Pick tau on dev (all dev sets pooled): the largest token reduction whose dev accuracy is within
    # max_drop of full reasoning; if none qualifies, the most conservative tau.
    dev_traces = sum((load_labeled(os.path.join(args.data, "labeled", f"{n}.jsonl")) for n in args.dev), [])
    dev_golds = gold_answers(dev_traces)
    dev_scores = dict(zip(keys_dev, predict(probe, X_dev)))
    dev_full = full_accuracy(dev_traces, dev_golds)
    dev_rows = {tau: score(simulate_early_exit(dev_traces, dev_scores, tau), dev_golds) for tau in args.tau}
    ok = [tau for tau, m in dev_rows.items() if m["accuracy"] >= dev_full - args.max_drop]
    best_tau = max(ok, key=lambda tau: dev_rows[tau]["token_reduction"]) if ok else max(args.tau)
    print(f"dev full accuracy {dev_full:.4f}; picked tau = {best_tau}")

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {"args": vars(args), "config": cfg, "grid": grid_results,
               "epochs": len(history), "best_dev_loss": min(history),
               "dev_full_accuracy": dev_full, "best_tau": best_tau,
               "dev": {str(tau): m for tau, m in dev_rows.items()}, "test": {}}
    for name in args.test:
        X, _, keys, _, _ = load_split(args.data, [name], mean, std)
        traces = load_labeled(os.path.join(args.data, "labeled", f"{name}.jsonl"))
        golds = gold_answers(traces)
        scores = dict(zip(keys, predict(probe, X)))
        labels = [int(c["correct"]) for t in traces for c in t["chunks"] if (t["id"], c["k"]) in scores]
        rows = {"full_accuracy": full_accuracy(traces, golds),
                "probe": probe_metrics([scores[(t["id"], c["k"])] for t in traces for c in t["chunks"]
                                        if (t["id"], c["k"]) in scores], labels)}
        print(f"\n{name}: full accuracy {rows['full_accuracy']:.4f}, probe AUC {rows['probe']['roc_auc']:.3f} "
              f"ECE {rows['probe']['ece']:.3f} Brier {rows['probe']['brier']:.3f}")
        for tau in args.tau:
            preds = simulate_early_exit(traces, scores, tau)
            m = score(preds, golds)
            rows[str(tau)] = {k: m[k] for k in ("accuracy", "avg_tokens", "token_reduction")}
            with open(os.path.join(args.out_dir, f"probe_{name}_tau{tau}.jsonl"), "w") as f:
                f.writelines(json.dumps(x) + "\n" for x in preds)
            mark = "  <- dev-picked" if tau == best_tau else ""
            print(f"  tau {tau:<5}  acc {m['accuracy']:.4f}  tokens {m['avg_tokens']:8.1f}  "
                  f"reduction {m['token_reduction']:.3f}{mark}")
        summary["test"][name] = rows
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
