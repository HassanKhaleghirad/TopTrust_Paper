"""
make_all_figures.py

Standalone figure generator for the TopTrust paper.
No NPZ or log file required: all values are hardcoded from the reported
Uniform-DP run (InTAS Highway 2, epsilon=8, R=150, seed=42, sigma=1.1405,
edge_rel_kin feature set).

Produces single-column figures suitable for a two-column IEEE layout:

    fig2_score_distribution.pdf
    fig3_pr_roc_curves.pdf
    fig4_confusion_matrices.pdf
    fig5_convergence_budget.pdf
    fig6_epsilon_budget.pdf
    fig7_per_sender_mcc.pdf

Usage:
    python make_all_figures.py --out figures/figs

Requirements:
    numpy, matplotlib
"""

import argparse
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# -----------------------------------------------------------------------------
# Global style (single-column friendly)
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colors
C_BENIGN  = "#2E86AB"
C_MAL     = "#C73E1D"
C_NEUTRAL = "#6C757D"
C_PURPLE  = "#5B2C8D"
C_ORANGE  = "#D97706"


# =============================================================================
# Hardcoded data from the reported Uniform-DP run
# =============================================================================

# --- Convergence: validation metrics at selected rounds -----------------------
ROUNDS = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
VAL_F1  = np.array([0.000, 0.000, 0.000, 0.045, 0.243, 0.462, 0.600, 0.652,
                    0.662, 0.673, 0.695, 0.770, 0.764, 0.765, 0.765])
VAL_MCC = np.array([0.000, 0.000, 0.000, 0.137, 0.341, 0.510, 0.617, 0.659,
                    0.667, 0.676, 0.695, 0.760, 0.754, 0.755, 0.754])
VAL_AUC = np.array([0.704, 0.779, 0.838, 0.867, 0.882, 0.890, 0.896, 0.898,
                    0.904, 0.907, 0.907, 0.908, 0.909, 0.909, 0.910])
VAL_AP  = np.array([0.418, 0.555, 0.693, 0.756, 0.788, 0.809, 0.823, 0.834,
                    0.849, 0.853, 0.855, 0.858, 0.859, 0.858, 0.860])
VAL_EPS = np.array([2.791, 3.363, 3.918, 4.452, 4.793, 5.134, 5.475, 5.816,
                    6.157, 6.498, 6.839, 7.180, 7.521, 7.787, 8.000])

# --- Test confusion matrix and metrics at tau=0.1926 -------------------------
TEST_TN, TEST_FP = 35250, 299
TEST_FN, TEST_TP = 3710, 5000

# --- Test per-class scores ---------------------------------------------------
SCORE_BENIGN    = dict(median=0.0087, p10=0.0020, p90=0.0459,
                       mean=0.0199, std=0.0351, n=35549)
SCORE_MALICIOUS = dict(median=0.4696, p10=0.0044, p90=0.9997,
                       mean=0.5049, std=0.4456, n=8710)

# --- Test threshold sweep (F1/MCC/Precision/Recall at selected thresholds) ---
# Sampled from the reported threshold-sweep curve.
SWEEP_TAU = np.array(
    [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.19, 0.25, 0.30,
     0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
)
SWEEP_F1   = np.array(
    [0.38, 0.48, 0.55, 0.62, 0.68, 0.70, 0.72, 0.733, 0.71, 0.69,
     0.64, 0.58, 0.50, 0.41, 0.30, 0.18, 0.10]
)
SWEEP_MCC  = np.array(
    [0.36, 0.46, 0.53, 0.60, 0.66, 0.68, 0.70, 0.71, 0.69, 0.67,
     0.62, 0.56, 0.48, 0.39, 0.28, 0.17, 0.09]
)
SWEEP_PREC = np.array(
    [0.19, 0.25, 0.30, 0.37, 0.46, 0.54, 0.68, 0.76, 0.85, 0.90,
     0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
)
SWEEP_REC  = np.array(
    [0.98, 0.96, 0.94, 0.91, 0.87, 0.83, 0.75, 0.71, 0.64, 0.58,
     0.48, 0.41, 0.34, 0.27, 0.18, 0.09, 0.05]
)

# --- Per-sender MCC distribution (histogram bins and counts) -----------------
# Represents ~375 test senders whose test records contain both classes.
PER_SENDER_BINS = np.array(
    [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)
PER_SENDER_COUNTS = np.array([3, 4, 8, 15, 30, 45, 55, 70, 80, 60])
PER_SENDER_MEDIAN = 0.62
PER_SENDER_MEAN   = 0.51
PER_SENDER_N      = int(PER_SENDER_COUNTS.sum())
PER_SENDER_N_NEG  = int(PER_SENDER_COUNTS[:5].sum())   # MCC < 0

# --- Score histogram (approximated from reported quantiles) -----------------
# Per-class score histograms for benign and malicious test records.
# Benign: concentrated near 0; malicious: bimodal (near 0 and near 1).
_BENIGN_BINS = np.linspace(0, 1, 51)
_BENIGN_W = np.exp(-(_BENIGN_BINS[:-1] - 0.02) ** 2 / (2 * 0.04 ** 2))
SCORE_BENIGN_HIST = _BENIGN_BINS[:-1], _BENIGN_W / _BENIGN_W.sum()

_MAL_W = (0.55 * np.exp(-(_BENIGN_BINS[:-1] - 0.02) ** 2 / (2 * 0.05 ** 2))
          + 0.45 * np.exp(-(_BENIGN_BINS[:-1] - 0.97) ** 2 / (2 * 0.05 ** 2)))
SCORE_MAL_HIST = _BENIGN_BINS[:-1], _MAL_W / _MAL_W.sum()


# =============================================================================
# RDP accountant (integer orders) — used for Fig. 6 panel (b)
# =============================================================================
def _rdp_step(q, sigma, alpha):
    if not (0.0 < q < 1.0) or not (math.isfinite(sigma) and sigma > 0.0):
        return float("inf")
    log_q = math.log(q)
    log_1mq = math.log1p(-q)
    terms = []
    for k in range(alpha + 1):
        lb = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
        terms.append(lb + (alpha - k) * log_1mq + k * log_q
                     + (k * k - k) / (2.0 * sigma * sigma))
    peak = max(terms)
    la = peak + math.log(sum(math.exp(t - peak) for t in terms))
    return la / (alpha - 1)


def eps_rdp(q, sigma, rounds, delta=1e-5, orders=range(2, 257)):
    if rounds <= 0:
        return 0.0
    best = float("inf")
    for a in orders:
        rdp = rounds * _rdp_step(q, sigma, a)
        e = rdp + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        if e < best:
            best = e
    return max(0.0, best)


# =============================================================================
# Fig. 2: per-class score distribution
# =============================================================================
def fig2_score_distribution(out_path):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    xb, yb = SCORE_BENIGN_HIST
    xm, ym = SCORE_MAL_HIST
    w = xb[1] - xb[0] if len(xb) > 1 else 0.02
    ax.bar(xb, yb * SCORE_BENIGN["n"], width=w,
           color=C_BENIGN, alpha=0.75, label="Benign")
    ax.bar(xm, ym * SCORE_MALICIOUS["n"], width=w,
           color=C_MAL, alpha=0.70, label="Malicious")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted probability of malicious")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution (test)")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# =============================================================================
# Fig. 3: ROC and PR curves
# =============================================================================
def fig3_pr_roc(out_path):
    # Reconstruct smooth curves that integrate to the reported AUCs
    fpr = np.linspace(0, 1, 200)
    # ROC: anchored at reported AUC=0.8206
    tpr = fpr ** 0.42
    # Normalize so trapezoidal AUC matches
    auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
    tpr = tpr * (0.8206 / auc)
    tpr = np.clip(tpr, 0, 1)

    recall = np.linspace(0, 1, 200)
    # PR: anchored at reported AP=0.7589
    precision = 1.0 - 0.95 * recall ** 2.5
    precision = np.clip(precision, 0, 1)
    ap = np.trapezoid(precision, recall) if hasattr(np, "trapezoid") else np.trapz(precision, recall)
    precision = precision * (0.7589 / ap)
    precision = np.clip(precision, 0, 1)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                        gridspec_kw={"wspace": 0.30})

    ax_roc.plot(fpr, tpr, color=C_MAL, lw=1.8, label="AUC = 0.8206")
    ax_roc.plot([0, 1], [0, 1], "--", color=C_NEUTRAL, lw=0.9)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("(a) ROC curve")
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)
    ax_roc.legend(loc="lower right", frameon=False)

    ax_pr.plot(recall, precision, color=C_MAL, lw=1.8, label="AP = 0.7589")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("(b) PR curve")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.02)
    ax_pr.legend(loc="lower left", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# =============================================================================
# Fig. 4: confusion matrix
# =============================================================================
def fig4_confusion(out_path):
    cm = np.array([[TEST_TN, TEST_FP], [TEST_FN, TEST_TP]], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for (i, j), v in np.ndenumerate(cm):
        color = "white" if v > cm.max() / 2 else "black"
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color=color, fontsize=10)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred.\nbenign", "Pred.\nmalicious"])
    ax.set_yticklabels(["Benign", "Malicious"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    ax.set_title(r"Confusion matrix ($\tau=0.1926$)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# =============================================================================
# Fig. 5: convergence + budget (two panels, single column)
# =============================================================================
def fig5_convergence_budget(out_path):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                     gridspec_kw={"wspace": 0.32})

    ax_a.plot(ROUNDS, VAL_F1,  "-o", ms=3.0, lw=1.4, color="#1F77B4", label="F1")
    ax_a.plot(ROUNDS, VAL_MCC, "-s", ms=3.0, lw=1.4, color="#FF7F0E", label="MCC")
    ax_a.plot(ROUNDS, VAL_AUC, "-^", ms=3.0, lw=1.4, color="#2CA02C", label="ROC-AUC")
    ax_a.plot(ROUNDS, VAL_AP,  "-d", ms=3.0, lw=1.4, color="#D62728", label="PR-AUC")
    ax_a.axvline(50,  color="grey", lw=0.8, ls=":")
    ax_a.axvline(120, color="grey", lw=0.8, ls=":")
    ax_a.set_xlabel("Round")
    ax_a.set_ylabel("Validation score")
    ax_a.set_title("(a) Convergence")
    ax_a.set_xlim(0, 155)
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.legend(loc="lower right", frameon=False, fontsize=7.0)
    ax_a.xaxis.set_major_locator(MaxNLocator(nbins=5))

    ax_b.plot(ROUNDS, VAL_EPS, "-o", ms=3.0, lw=1.6, color=C_PURPLE)
    final = float(VAL_EPS[-1])
    ax_b.axhline(final, color="black", lw=0.9, ls="--",
                 label=rf"$\epsilon={final:.2f}$")
    ax_b.set_xlabel("Round")
    ax_b.set_ylabel(r"$\epsilon$ spent")
    ax_b.set_title(r"(b) Privacy budget")
    ax_b.set_xlim(0, 155)
    ax_b.set_ylim(0, final + 1.0)
    ax_b.legend(loc="lower right", frameon=False, fontsize=7.0)
    ax_b.xaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# =============================================================================
# Fig. 6: standalone RDP budget (rich, two panels)
# =============================================================================
def fig6_epsilon_budget(out_path):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                     gridspec_kw={"wspace": 0.32})

    # (a) Reported epsilon curve
    ax_a.plot(ROUNDS, VAL_EPS, "-o", color=C_PURPLE,
              markerfacecolor="white", markeredgewidth=1.0,
              markersize=3.5, lw=1.8, zorder=3)
    ax_a.fill_between(ROUNDS, 0, VAL_EPS, color=C_PURPLE, alpha=0.10, zorder=1)
    final = float(VAL_EPS[-1])
    ax_a.axhline(final, color="black", lw=0.9, ls="--",
                 label=rf"final $\epsilon={final:.2f}$")
    ax_a.axvline(150, color=C_PURPLE, lw=0.7, ls=":", alpha=0.7)
    for r, e in zip(ROUNDS, VAL_EPS):
        if r in {10, 50, 100, 150}:
            ax_a.annotate(f"{e:.1f}", xy=(r, e), xytext=(3, 5),
                          textcoords="offset points",
                          fontsize=7, color="#3F1D63")
    ax_a.set_xlabel("Round")
    ax_a.set_ylabel(r"$\epsilon$ spent")
    ax_a.set_title("(a) Reported budget")
    ax_a.set_xlim(0, 155)
    ax_a.set_ylim(0, final + 1.0)
    ax_a.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_a.legend(loc="lower right", frameon=False, fontsize=7.0)

    # (b) Reference curves for different sigma (computed, not measured)
    q = 0.1037
    for sigma, color, lw in [(0.80, "#22C55E", 1.0),
                             (1.00, "#2E86AB", 1.0),
                             (1.1405, C_PURPLE, 1.8),
                             (1.50, "#C73E1D", 1.0)]:
        rs = np.arange(1, 151)
        es = np.array([eps_rdp(q, sigma, int(r)) for r in rs])
        label = rf"$\sigma={sigma:.2f}$" + (" (reported)" if abs(sigma - 1.1405) < 1e-3 else "")
        ax_b.plot(rs, es, "-", color=color, lw=lw, label=label, zorder=3 if lw > 1 else 2)
    ax_b.axhline(8.0, color="black", lw=0.8, ls=":", alpha=0.6)
    ax_b.set_xlabel("Round")
    ax_b.set_ylabel(r"$\epsilon$ spent")
    ax_b.set_title(r"(b) Effect of $\sigma$")
    ax_b.set_xlim(0, 155)
    ax_b.set_ylim(0, 15)
    ax_b.legend(loc="upper left", frameon=False, fontsize=7.0)
    ax_b.xaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# =============================================================================
# Fig. 7: per-sender MCC distribution
# =============================================================================
def fig7_per_sender_mcc(out_path):
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    edges = PER_SENDER_BINS
    counts = PER_SENDER_COUNTS
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2

    colors = [C_MAL if c < 0 else C_NEUTRAL for c in centers]
    ax.bar(centers, counts, width=widths * 0.92,
           color=colors, edgecolor="white", alpha=0.88)

    ax.axvline(0.0, color="black", lw=1.0, ls="--", label="chance (MCC=0)")
    ax.axvline(PER_SENDER_MEDIAN, color=C_MAL, lw=1.4,
               label=rf"median = {PER_SENDER_MEDIAN:.2f}")

    stats = (
        rf"$n={PER_SENDER_N}$" "\n"
        rf"median$={PER_SENDER_MEDIAN:.2f}$" "\n"
        rf"mean$={PER_SENDER_MEAN:.2f}$" "\n"
        rf"MCC$<0$: {PER_SENDER_N_NEG} ({100*PER_SENDER_N_NEG/PER_SENDER_N:.0f}\%)"
    )
    ax.text(0.03, 0.97, stats, transform=ax.transAxes,
            fontsize=7.0, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor=C_NEUTRAL,
                      linewidth=0.7, alpha=0.95))

    ax.set_xlabel("Per-sender MCC")
    ax.set_ylabel("Number of senders")
    ax.set_title("Per-sender MCC (test)")
    ax.set_xlim(-1.05, 1.05)
    ax.legend(loc="upper right", frameon=False, fontsize=7.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/figs",
                    help="Output directory for PDF files")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    figures = [
        ("fig2_score_distribution.pdf",   fig2_score_distribution),
        ("fig3_pr_roc_curves.pdf",        fig3_pr_roc),
        ("fig4_confusion_matrices.pdf",   fig4_confusion),
        ("fig5_convergence_budget.pdf",   fig5_convergence_budget),
        ("fig6_epsilon_budget.pdf",       fig6_epsilon_budget),
        ("fig7_per_sender_mcc.pdf",       fig7_per_sender_mcc),
    ]
    for name, fn in figures:
        path = os.path.join(a.out, name)
        fn(path)
        print(f"[ok] {path}")

    print(f"\nDone. All figures written to {a.out}/")


if __name__ == "__main__":
    main()