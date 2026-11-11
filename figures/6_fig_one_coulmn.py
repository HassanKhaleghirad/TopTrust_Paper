
import argparse
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# -----------------------------------------------------------------------------
# Reported run: R=150, q=0.1037, sigma=1.1405, delta=1e-5
# These are the epsilon values printed by the training log.
# -----------------------------------------------------------------------------
REPORTED_ROUNDS = np.array(
    [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
)
REPORTED_EPS = np.array(
    [2.791, 3.363, 3.918, 4.452, 4.793, 5.134, 5.475, 5.816, 6.157,
     6.498, 6.839, 7.180, 7.521, 7.787, 8.000]
)

Q_DEFAULT = 0.1037
SIGMA_DEFAULT = 1.1405
DELTA_DEFAULT = 1e-5
ROUNDS_DEFAULT = 150


def make_figure_single(out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(REPORTED_ROUNDS, REPORTED_EPS,
            "-o", color="#5B2C8D", markerfacecolor="white",
            markeredgewidth=1.4, markersize=5.5, lw=2.2, zorder=3)
    ax.fill_between(REPORTED_ROUNDS, 0, REPORTED_EPS,
                    color="#5B2C8D", alpha=0.12, zorder=1)
    final_eps = float(REPORTED_EPS[-1])
    ax.axhline(final_eps, color="black", lw=1.1, ls="--",
               label=rf"final $\epsilon = {final_eps:.2f}$ at round 150")
    ax.axvline(150, color="#5B2C8D", lw=0.9, ls=":", alpha=0.7)
    # Annotate a few points
    for r, e in zip(REPORTED_ROUNDS, REPORTED_EPS):
        if r in {10, 50, 100, 150}:
            ax.annotate(f"{e:.2f}", xy=(r, e), xytext=(4, 6),
                        textcoords="offset points", fontsize=8,
                        color="#3F1D63")
    # Shaded regions
    ax.axhspan(0, 1.0, color="#2E86AB", alpha=0.06)
    ax.axhspan(8.0, final_eps + 0.5, color="#C73E1D", alpha=0.06)
    ax.set_xlabel("Communication round")
    ax.set_ylabel(r"Epsilon spent ($\epsilon$)")
    ax.set_title(r"RDP-composed privacy budget (Uniform-DP, $\sigma=1.14$)")
    ax.set_xlim(0, 155)
    ax.set_ylim(0, final_eps + 1.0)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig6-single] wrote {out_path}")