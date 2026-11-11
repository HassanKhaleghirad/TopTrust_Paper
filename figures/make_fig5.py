"""
make_fig5.py

Generate the three figures requested:

  fig5_convergence_budget.pdf
      Two-panel figure:
        (a) Validation F1, MCC, ROC-AUC, PR-AUC over communication rounds.
        (b) RDP-composed privacy budget over the same rounds.

  fig_epsilon_budget_standalone.pdf
      Standalone RDP budget curve (larger, for slide decks or a dedicated
      figure in the paper).

  fig_per_sender_mcc.pdf
      Distribution of per-sender MCC on the test partition,
      computed from the NPZ. Only senders whose test records contain
      both benign and malicious messages are included.

Usage:
    python make_fig5.py \
        --npz uniform_dp_predictions_eps8.0_seed42_rounds150_cmax1.0_edge_rel_kin.npz \
        --out figs/ \
        --threshold 0.1926

    # If the training log is available, convergence values are read from it:
    python make_fig5.py --npz ... --log train_log.txt --out figs/

    # If the log is not available, the hardcoded values from the reported run
    # (R=150, eta=0.15, sigma=1.1405, seed=42) are used.
"""

import argparse
import math
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# -----------------------------------------------------------------------------
# Hardcoded convergence values from the reported run
# (R=150, eta=0.15, C=1.0, sigma=1.1405, seed=42)
# Used only when the training log is not available.
# -----------------------------------------------------------------------------
FALLBACK_ROUNDS = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
FALLBACK_F1   = np.array([0.000, 0.000, 0.000, 0.045, 0.243, 0.462, 0.600,
                          0.652, 0.662, 0.673, 0.695, 0.770, 0.764, 0.765, 0.765])
FALLBACK_MCC  = np.array([0.000, 0.000, 0.000, 0.137, 0.341, 0.510, 0.617,
                          0.659, 0.667, 0.676, 0.695, 0.760, 0.754, 0.755, 0.754])
FALLBACK_AUC  = np.array([0.704, 0.779, 0.838, 0.867, 0.882, 0.890, 0.896,
                          0.898, 0.904, 0.907, 0.907, 0.908, 0.909, 0.909, 0.910])
FALLBACK_AP   = np.array([0.418, 0.555, 0.693, 0.756, 0.788, 0.809, 0.823,
                          0.834, 0.849, 0.853, 0.855, 0.858, 0.859, 0.858, 0.860])
FALLBACK_EPS  = np.array([2.791, 3.363, 3.918, 4.452, 4.793, 5.134, 5.475,
                          5.816, 6.157, 6.498, 6.839, 7.180, 7.521, 7.787, 8.000])


# -----------------------------------------------------------------------------
# Parse the training log, if available
# -----------------------------------------------------------------------------
LOG_PATTERN = re.compile(
    r"round\s+(?P<round>\d+)\s+"
    r"val F1=(?P<f1>[0-9.]+)\s+"
    r"MCC=(?P<mcc>[0-9.]+)\s+"
    r"AUC=(?P<auc>[0-9.]+)\s+"
    r"AP=(?P<ap>[0-9.]+)\s+"
    r"lambda=(?P<lam>[0-9.]+)\s+"
    r"eps=(?P<eps>[0-9.]+)"
)


def parse_log(path):
    if not path or not os.path.exists(path):
        return None
    rounds, f1s, mccs, aucs, aps, eps = [], [], [], [], [], []
    with open(path, "r") as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if not m:
                continue
            rounds.append(int(m.group("round")))
            f1s.append(float(m.group("f1")))
            mccs.append(float(m.group("mcc")))
            aucs.append(float(m.group("auc")))
            aps.append(float(m.group("ap")))
            eps.append(float(m.group("eps")))
    if not rounds:
        return None
    return dict(rounds=np.array(rounds),
                f1=np.array(f1s), mcc=np.array(mccs),
                auc=np.array(aucs), ap=np.array(aps),
                eps=np.array(eps))


def load_convergence_data(log_path):
    parsed = parse_log(log_path)
    if parsed is not None:
        print(f"[log] parsed {len(parsed['rounds'])} round entries from {log_path}")
        return parsed
    print("[log] not found; using hardcoded values from the reported run")
    return dict(rounds=FALLBACK_ROUNDS, f1=FALLBACK_F1, mcc=FALLBACK_MCC,
                auc=FALLBACK_AUC, ap=FALLBACK_AP, eps=FALLBACK_EPS)


# -----------------------------------------------------------------------------
# Fig 5: two-panel convergence + budget
# -----------------------------------------------------------------------------
def fig5_convergence_and_budget(data, out_path):
    fig, (ax_conv, ax_budget) = plt.subplots(1, 2, figsize=(11, 4))

    # Panel (a): validation metrics
    ax_conv.plot(data["rounds"], data["f1"], "-o", ms=4, lw=1.8,
                 color="#1F77B4", label="F1")
    ax_conv.plot(data["rounds"], data["mcc"], "-s", ms=4, lw=1.8,
                 color="#FF7F0E", label="MCC")
    ax_conv.plot(data["rounds"], data["auc"], "-^", ms=4, lw=1.8,
                 color="#2CA02C", label="ROC-AUC")
    ax_conv.plot(data["rounds"], data["ap"], "-d", ms=4, lw=1.8,
                 color="#D62728", label="PR-AUC")
    ax_conv.axvline(50, color="grey", lw=1, ls=":")
    ax_conv.axvline(120, color="grey", lw=1, ls=":")
    ax_conv.text(50, 0.04, " 50", fontsize=8, color="grey", va="bottom")
    ax_conv.text(120, 0.04, " 120", fontsize=8, color="grey", va="bottom")
    ax_conv.set_xlabel("Communication round")
    ax_conv.set_ylabel("Validation score")
    ax_conv.set_title("(a) Validation metrics")
    ax_conv.legend(loc="lower right", frameon=False)
    ax_conv.set_xlim(0, data["rounds"].max() + 5)
    ax_conv.set_ylim(-0.02, 1.02)

    # Panel (b): privacy budget
    ax_budget.plot(data["rounds"], data["eps"], "-o", ms=4, lw=2,
                   color="#9467BD")
    final_eps = float(data["eps"][-1])
    ax_budget.axhline(final_eps, color="black", lw=1, ls="--",
                      label=f"final $\\epsilon = {final_eps:.2f}$")
    ax_budget.set_xlabel("Communication round")
    ax_budget.set_ylabel("Epsilon spent (RDP accountant)")
    ax_budget.set_title("(b) RDP privacy budget")
    ax_budget.legend(loc="lower right", frameon=False)
    ax_budget.set_xlim(0, data["rounds"].max() + 5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig5] {out_path}")


# -----------------------------------------------------------------------------
# Standalone RDP budget
# -----------------------------------------------------------------------------
def fig_epsilon_budget_standalone(data, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(data["rounds"], data["eps"], "-o", ms=5, lw=2.2,
            color="#9467BD", markerfacecolor="white", markeredgewidth=1.5)
    final_eps = float(data["eps"][-1])
    ax.axhline(final_eps, color="black", lw=1.2, ls="--",
               label=f"final $\\epsilon = {final_eps:.2f}$")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Epsilon spent (RDP accountant)")
    ax.set_title("RDP-composed privacy budget\n(Uniform-DP, $\\sigma=1.14$, $q=0.104$)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, data["rounds"].max() + 5)
    ax.set_ylim(0, final_eps + 1.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[eps]  {out_path}")


# -----------------------------------------------------------------------------
# Per-sender MCC
# -----------------------------------------------------------------------------
def metrics_binary(y, p, threshold):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-12))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return mcc


def fig_per_sender_mcc(sender_id, y, p, threshold, out_path):
    by_sender = defaultdict(lambda: ([], []))
    for s, yi, pi in zip(sender_id, y, p):
        by_sender[s][0].append(int(yi))
        by_sender[s][1].append(float(pi))

    mccs, counts = [], []
    for s, (yy, pp) in by_sender.items():
        yy = np.asarray(yy)
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        pp = np.asarray(pp)
        mccs.append(metrics_binary(yy, pp, threshold))
        counts.append(len(yy))

    if not mccs:
        print("[per-sender] no sender with both classes; figure skipped")
        return

    mccs = np.array(mccs)
    n = len(mccs)
    n_neg = int((mccs < 0).sum())
    print(f"[per-sender] n={n} senders, {n_neg} with MCC < 0, "
          f"median MCC={np.median(mccs):.3f}")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.linspace(-1.0, 1.0, 41)
    ax.hist(mccs, bins=bins, color="#6C757D", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", lw=1.2, ls="--", label="MCC = 0 (chance)")
    ax.axvline(np.median(mccs), color="#C73E1D", lw=1.5, ls="-",
               label=f"median = {np.median(mccs):.3f}")
    ax.set_xlabel("Per-sender MCC")
    ax.set_ylabel("Number of senders")
    ax.set_title(
        f"Per-sender MCC on the test partition\n"
        f"({n} senders with both benign and malicious test messages; "
        f"{n_neg} with MCC $< 0$)"
    )
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[mcc]  {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz",
        default="uniform_dp_predictions_eps8.0_seed42_rounds150_cmax1.0_edge_rel_kin.npz",
        help="NPZ produced by Edited_toptrust_version4.py --mode uniform",
    )
    ap.add_argument(
        "--log",
        default=None,
        help="Optional training log. If absent, hardcoded values from the reported run are used.",
    )
    ap.add_argument("--out", default="figs", help="Output directory")
    ap.add_argument("--threshold", type=float, default=0.1926,
                    help="Validation-selected decision threshold")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)

    # -----------------------------------------------------------------
    # 1) Convergence + budget (fig 5)
    # -----------------------------------------------------------------
    data = load_convergence_data(a.log)
    fig5_convergence_and_budget(
        data, os.path.join(a.out, "fig5_convergence_budget.pdf")
    )
    fig_epsilon_budget_standalone(
        data, os.path.join(a.out, "fig_epsilon_budget_standalone.pdf")
    )

    # -----------------------------------------------------------------
    # 2) Per-sender MCC (from NPZ)
    # -----------------------------------------------------------------
    if not os.path.exists(a.npz):
        print(f"[npz] not found: {a.npz}")
        print("      per-sender figure skipped")
        return

    d = np.load(a.npz, allow_pickle=False)
    if "sender_id" not in d.files:
        print(f"[npz] no sender_id field in {a.npz}; per-sender figure skipped")
        return

    y = d["y_test"].astype(np.int64)
    p = d["test_prob"].astype(np.float64)
    sender_id = d["sender_id"].astype(str)

    fig_per_sender_mcc(
        sender_id, y, p, a.threshold,
        os.path.join(a.out, "fig_per_sender_mcc.pdf"),
    )


if __name__ == "__main__":
    main()