"""
make_fig6_budget_rich.py

Rich three-panel figure for the RDP-composed privacy budget of the
Uniform-DP baseline.

Panel (a): epsilon spent vs communication round, with the consumed budget
           shaded and key rounds annotated.
Panel (b): marginal epsilon spent per round (derivative of panel a),
           showing that the per-round cost is nearly constant (the small
           decrease reflects the RDP-to-(eps,delta) conversion becoming
           tighter as the accumulated budget grows).
Panel (c): reference curves for other noise multipliers sigma, computed
           with the same integer-order RDP accountant used by the model.
           The reported configuration is highlighted.

Usage:
    python make_fig6_budget_rich.py --out figs/fig6_epsilon_budget.pdf
"""

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


# -----------------------------------------------------------------------------
# RDP accountant (integer orders, matches the model code)
# -----------------------------------------------------------------------------
def _rdp_step(q, sigma, alpha):
    if not (0.0 < q < 1.0) or not (math.isfinite(sigma) and sigma > 0.0):
        return float("inf")
    log_q = math.log(q)
    log_1_minus_q = math.log1p(-q)
    terms = []
    for k in range(alpha + 1):
        log_binom = (math.lgamma(alpha + 1) - math.lgamma(k + 1)
                     - math.lgamma(alpha - k + 1))
        terms.append(log_binom
                     + (alpha - k) * log_1_minus_q
                     + k * log_q
                     + (k * k - k) / (2.0 * sigma * sigma))
    peak = max(terms)
    log_a = peak + math.log(sum(math.exp(t - peak) for t in terms))
    return log_a / (alpha - 1)


def epsilon_rdp(q, sigma, rounds, delta, orders=range(2, 257)):
    if rounds <= 0:
        return 0.0
    best = float("inf")
    for alpha in orders:
        rdp = rounds * _rdp_step(q, sigma, alpha)
        eps = (rdp
               + math.log((alpha - 1) / alpha)
               - (math.log(delta) + math.log(alpha)) / (alpha - 1))
        if eps < best:
            best = eps
    return max(0.0, best)


def epsilon_curve(q, sigma, max_rounds, delta):
    rounds = np.arange(1, max_rounds + 1)
    eps = np.array([epsilon_rdp(q, sigma, r, delta) for r in rounds])
    return rounds, eps


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
def make_figure(out_path):
    fig = plt.figure(figsize=(12.5, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0],
                          wspace=0.32)

    # -----------------------------------------------------------------
    # Panel (a): reported epsilon curve
    # -----------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(REPORTED_ROUNDS, REPORTED_EPS,
              "-o", color="#5B2C8D", markerfacecolor="white",
              markeredgewidth=1.4, markersize=5, lw=2.0, zorder=3)
    ax_a.fill_between(REPORTED_ROUNDS, 0, REPORTED_EPS,
                      color="#5B2C8D", alpha=0.10, zorder=1)

    # Final epsilon
    final_eps = float(REPORTED_EPS[-1])
    final_round = int(REPORTED_ROUNDS[-1])
    ax_a.axhline(final_eps, color="black", lw=1.0, ls="--",
                 label=rf"final $\epsilon = {final_eps:.2f}$ at round {final_round}")
    ax_a.axvline(final_round, color="#5B2C8D", lw=0.9, ls=":", alpha=0.7)

    # Annotate selected points only (avoid clutter)
    annotate_rounds = {10, 50, 100, 150}
    for r, e in zip(REPORTED_ROUNDS, REPORTED_EPS):
        if r in annotate_rounds:
            ax_a.annotate(
                f"{e:.2f}",
                xy=(r, e), xytext=(4, 6),
                textcoords="offset points",
                fontsize=7.5, color="#3F1D63",
            )

    # Non-private / private region shading
    ax_a.axhspan(0, 1.0, color="#2E86AB", alpha=0.05)
    ax_a.axhspan(8.0, final_eps + 0.5, color="#C73E1D", alpha=0.05)

    ax_a.set_xlabel("Communication round")
    ax_a.set_ylabel(r"Epsilon spent ($\epsilon$)")
    ax_a.set_title("(a) Reported privacy budget")
    ax_a.set_xlim(0, final_round + 5)
    ax_a.set_ylim(0, final_eps + 1.0)
    ax_a.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_a.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_a.legend(loc="lower right", frameon=False)

    # -----------------------------------------------------------------
    # Panel (b): marginal cost per round
    # -----------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    marginal = np.diff(np.concatenate(([0.0], REPORTED_EPS)))
    marginal = marginal / np.diff(np.concatenate(([0.0], REPORTED_ROUNDS)))
    ax_b.plot(REPORTED_ROUNDS, marginal,
              "-s", color="#D97706", markerfacecolor="white",
              markeredgewidth=1.3, markersize=4.5, lw=1.8)
    ax_b.fill_between(REPORTED_ROUNDS, 0, marginal,
                      color="#D97706", alpha=0.18)
    mean_cost = float(REPORTED_EPS[-1] / REPORTED_ROUNDS[-1])
    ax_b.axhline(mean_cost, color="black", lw=1.0, ls="--",
                 label=rf"mean cost = {mean_cost:.4f} / round")
    ax_b.set_xlabel("Communication round")
    ax_b.set_ylabel(r"$\partial \epsilon / \partial r$")
    ax_b.set_title("(b) Marginal cost per round")
    ax_b.set_xlim(0, final_round + 5)
    ax_b.set_ylim(0, max(marginal) * 1.25)
    ax_b.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_b.legend(loc="lower right", frameon=False)

    # -----------------------------------------------------------------
    # Panel (c): reference curves for different sigma
    # -----------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    q, delta, R = Q_DEFAULT, DELTA_DEFAULT, ROUNDS_DEFAULT
    for sigma, color, label in [
        (0.80, "#22C55E", r"$\sigma=0.80$"),
        (1.00, "#2E86AB", r"$\sigma=1.00$"),
        (SIGMA_DEFAULT, "#5B2C8D", rf"$\sigma={SIGMA_DEFAULT:.2f}$ (reported)"),
        (1.50, "#C73E1D", r"$\sigma=1.50$"),
    ]:
        rounds_c, eps_c = epsilon_curve(q, sigma, R, delta)
        is_reported = abs(sigma - SIGMA_DEFAULT) < 1e-3
        ax_c.plot(rounds_c, eps_c, "-",
                  color=color, lw=2.2 if is_reported else 1.4,
                  label=label, zorder=3 if is_reported else 2)
        # Mark the final value
        ax_c.plot(rounds_c[-1], eps_c[-1], "o",
                  color=color, markersize=4.5, zorder=4)

    ax_c.axhline(8.0, color="black", lw=0.9, ls=":", alpha=0.6)
    ax_c.set_xlabel("Communication round")
    ax_c.set_ylabel(r"Epsilon spent ($\epsilon$)")
    ax_c.set_title(r"(c) Reference curves for $\sigma$")
    ax_c.set_xlim(0, R + 5)
    ax_c.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_c.legend(loc="upper left", frameon=False, fontsize=7.6)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig6] wrote {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/fig6_epsilon_budget.pdf")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    make_figure(a.out)


if __name__ == "__main__":
    main()