"""
Plot accuracy vs. token reduction on each test set, for the probes and the simple baselines.

Reads results/probe_{linear,mlp}/summary.json (test accuracy and token reduction per tau) and
results/simple/*.metrics.json (full, first, budget N). Writes one figure with a panel per test set
to results/figures/tradeoff.png (and .pdf).

Usage:
    python results/plot_tradeoff.py
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = [("gsm8k_test", "GSM8K"), ("math500_test", "MATH-500"), ("aime2024_test", "AIME 2024")]
PROBES = [("probe_linear", "probe (linear)", "tab:blue", "o"), ("probe_mlp", "probe (MLP)", "tab:orange", "s")]


def load(path):
    with open(path) as f:
        return json.load(f)


def simple_metrics(name):
    """{rule: metrics} for one test set, e.g. {"full": {...}, "first": {...}, "budget1024": {...}}."""
    out = {}
    for path in glob.glob(os.path.join(HERE, "simple", f"*_{name}.metrics.json")):
        rule = os.path.basename(path)[: -len(f"_{name}.metrics.json")]
        out[rule] = load(path)
    return out


def main():
    fig, axes = plt.subplots(1, len(SETS), figsize=(5 * len(SETS), 4.2))
    for ax, (name, title) in zip(axes, SETS):
        simple = simple_metrics(name)

        # Probes: one point per tau, joined in tau order.
        for folder, label, color, marker in PROBES:
            path = os.path.join(HERE, folder, "summary.json")
            if not os.path.exists(path):
                continue
            s = load(path)
            test = s["test"][name]
            taus = sorted(k for k in test if re.fullmatch(r"[0-9.]+", k))
            taus = sorted(taus, key=float)
            x = [100 * test[t]["token_reduction"] for t in taus]
            y = [100 * test[t]["accuracy"] for t in taus]
            ax.plot(x, y, marker=marker, color=color, label=label, ms=4)
            # Mark the tau picked on dev.
            best = str(s["best_tau"])
            if best in test:
                ax.scatter(100 * test[best]["token_reduction"], 100 * test[best]["accuracy"], s=120,
                           facecolors="none", edgecolors=color, linewidths=1.5, zorder=5)

        # Fixed token budgets: one point per N, joined in N order.
        budgets = sorted((int(r[6:]), m) for r, m in simple.items() if re.fullmatch(r"budget\d+", r))
        if budgets:
            ax.plot([100 * m["token_reduction"] for _, m in budgets], [100 * m["accuracy"] for _, m in budgets],
                    "--^", color="tab:green", label="token budget N", ms=5)
            for n, m in budgets:
                ax.annotate(str(n), (100 * m["token_reduction"], 100 * m["accuracy"]), fontsize=7,
                            textcoords="offset points", xytext=(4, -9), color="tab:green")

        # Single-point baselines.
        if "full" in simple:
            ax.axhline(100 * simple["full"]["accuracy"], color="gray", ls=":", lw=1)
            ax.scatter(0, 100 * simple["full"]["accuracy"], color="black", marker="*", s=90, zorder=6,
                       label="full reasoning")
        if "first" in simple:
            m = simple["first"]
            ax.scatter(100 * m["token_reduction"], 100 * m["accuracy"], color="tab:red", marker="D", s=40,
                       zorder=6, label="first answer")

        ax.set_title(f"{title} (n = {simple.get('full', {}).get('n', '?')})")
        ax.set_xlabel("token reduction vs. full reasoning (%)")
        ax.set_ylabel("accuracy (%)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Accuracy vs. tokens saved (circled: tau picked on dev)", fontsize=11)
    fig.tight_layout()

    out_dir = os.path.join(HERE, "figures")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"tradeoff.{ext}"), dpi=200)
    print("wrote", os.path.join(out_dir, "tradeoff.png"), "and .pdf")


if __name__ == "__main__":
    main()
