"""
make_fig6.py

Fig. 6: RDP-composed privacy budget over communication rounds for the
reported Uniform-DP baseline. The final value is epsilon = 8.00 at round 150.

Standalone figure intended for direct inclusion in the paper.

Usage:
    python make_fig6.py --out figs/fig6_epsilon_budget.pdf
    python make_fig6.py --log train_log.txt --out figs/fig6_epsilon_budget.pdf
"""

import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid": True,
    "grid.alpha": 0.30,
})


# -----------------------------------------------------------------------------
# Fallback values from the reported run
# (R=150, eta=0.15, C=1.0, sigma=1.1405, seed=42)
# -----------------------------------------------------------------------------
FALLBACK_ROUNDS = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
FALLBACK_EPS    = np.array([2.791, 3.363, 3.918, 4.452, 4.793, 5.134, 5.475,
                            5.816, 6.157, 6.498, 6.839, 7.180, 7.521, 7.787, 8.000])


# -----------------------------------------------------------------------------
# Parse the training log, if available
# -----------------------------------------------------------------------------
LOG_PATTERN = re.compile(
    r"round\s+(?P<round>\d+)\s+.*?eps=(?P<eps>[0-9.]+)"
)


def parse_log_eps(path):
    """Return (rounds, eps) arrays parsed from a training log, or None."""
    if not path or not os.path.exists(path):
        return None
    rounds, eps = [], []
    with open(path, "r") as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if not m:
                continue
            rounds.append(int(m.group("round")))
            eps.append(float(m.group("eps")))
    if not rounds:
        return None
    return np.array(rounds), np.array(eps)


def load_budget(log_path=None):
    parsed = parse_log_eps(log_path)
    if parsed is not None:
        rounds, eps = parsed
        print(f"[log] parsed {len(rounds)} epsilon entries from {log_path}")
        return rounds, eps
    print("[log] not provided or unparsable; using fallback values from the reported run")
    return FALLBACK_ROUNDS, FALLBACK_EPS


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
def make_fig6(rounds, eps, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # Privacy budget curve
    ax.plot(
        rounds, eps,
        "-o",
        color="#5B2C8D",
        markerfacecolor="white",
        markeredgecolor="#5B2C8D",
        markeredgewidth=1.4,
        markersize=5.5,
        linewidth=2.2,
        label=r"$\epsilon$ spent",
        zorder=3,
    )

    # Fill under the curve
    ax.fill_between(rounds, 0, eps, color="#5B2C8D", alpha=0.10, zorder=1)

    # Final epsilon horizontal guide
    final_eps = float(eps[-1])
    ax.axhline(
        final_eps,
        color="black",
        linewidth=1.1,
        linestyle="--",
        zorder=2,
        label=rf"final $\epsilon = {final_eps:.2f}$ at round {int(rounds[-1])}",
    )

    # Vertical marker at the final round
    ax.axvline(
        rounds[-1],
        color="#5B2C8D",
        linewidth=0.9,
        linestyle=":",
        alpha=0.8,
        zorder=2,
    )

    ax.set_xlabel("Communication round")
    ax.set_ylabel(r"Epsilon spent ($\epsilon$)")
    ax.set_title(
        "RDP-composed privacy budget\n"
        r"(Uniform-DP baseline: $\sigma=1.14$, $q=0.104$, $\delta=10^{-5}$)"
    )

    ax.set_xlim(0, rounds.max() * 1.05)
    ax.set_ylim(0, final_eps + 1.0)

    # Annotate every other point on the curve (readability)
    for r, e in zip(rounds[::2], eps[::2]):
        ax.annotate(
            f"{e:.2f}",
            xy=(r, e),
            xytext=(3, 6),
            textcoords="offset points",
            fontsize=7.5,
            color="#3F1D63",
        )

    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig6] wrote {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None,
                    help="Optional training log; if absent, fallback values are used.")
    ap.add_argument("--out", default="figs/fig6_epsilon_budget.pdf",
                    help="Output PDF path")
    a = ap.parse_args()

    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rounds, eps = load_budget(a.log)
    make_fig6(rounds, eps, a.out)


if __name__ == "__main__":
    main()