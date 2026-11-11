"""
make_fig4_and_7.py

Generate two distinct figures:

  Fig. 4 : Threshold sweep on the test partition. The Validation-selected
           threshold is marked with a dotted vertical line.

  Fig. 7 : Distribution of per-sender MCC on the test partition. Only senders
           with both benign and malicious test messages are included. The
           dashed line marks MCC = 0 (chance).

Usage:
    python make_fig4_and_7.py \
        --npz uniform_dp_predictions_eps8.0_seed42_rounds150_cmax1.0_edge_rel_kin.npz \
        --out figs/ \
        --threshold 0.1926
"""

import argparse
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    "grid.alpha": 0.30,
})


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------
def binary_metrics(y, p, threshold):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-12))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=prec, recall=rec, f1=f1, mcc=mcc)


# -----------------------------------------------------------------------------
# Fig. 4: threshold sweep
# -----------------------------------------------------------------------------
def fig4_threshold_sweep(y, p, selected_threshold, out_path):
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s, mccs, precs, recs = [], [], [], []
    for t in thresholds:
        m = binary_metrics(y, p, t)
        f1s.append(m["f1"])
        mccs.append(m["mcc"])
        precs.append(m["precision"])
        recs.append(m["recall"])

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(thresholds, f1s,   lw=2.0, color="#1F77B4", label="F1")
    ax.plot(thresholds, mccs,  lw=2.0, color="#FF7F0E", label="MCC")
    ax.plot(thresholds, precs, lw=1.6, ls="--", color="#2CA02C", label="Precision")
    ax.plot(thresholds, recs,  lw=1.6, ls="--", color="#D62728", label="Recall")

    # Validation-selected threshold (dotted vertical line, per caption)
    ax.axvline(selected_threshold, color="black", lw=1.3, ls=":",
               label=rf"Validation-selected $\tau = {selected_threshold:.4f}$")

    # Annotate the F1 value at the selected threshold
    sel_metrics = binary_metrics(y, p, selected_threshold)
    ax.plot([selected_threshold], [sel_metrics["f1"]], "o",
            color="black", ms=6, markerfacecolor="white", markeredgewidth=1.4)
    ax.annotate(
        rf"F1$={sel_metrics['f1']:.3f}$",
        xy=(selected_threshold, sel_metrics["f1"]),
        xytext=(selected_threshold + 0.04, sel_metrics["f1"] + 0.06),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="black"),
    )

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sweep on the test partition")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="center right", frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig4] wrote {out_path}")


# -----------------------------------------------------------------------------
# Fig. 7: per-sender MCC distribution
# -----------------------------------------------------------------------------
def fig7_per_sender_mcc(sender_id, y, p, threshold, out_path):
    by_sender = defaultdict(lambda: ([], []))
    for s, yi, pi in zip(sender_id, y, p):
        by_sender[s][0].append(int(yi))
        by_sender[s][1].append(float(pi))

    mccs = []
    n_total = len(by_sender)
    for s, (yy, pp) in by_sender.items():
        yy = np.asarray(yy)
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        pp = np.asarray(pp)
        m = binary_metrics(yy, pp, threshold)
        mccs.append(m["mcc"])

    if not mccs:
        print("[fig7] no sender with both classes; figure skipped")
        return

    mccs = np.array(mccs)
    n_used = len(mccs)
    n_neg = int((mccs < 0).sum())
    median_mcc = float(np.median(mccs))
    mean_mcc = float(np.mean(mccs))

    print(f"[fig7] used {n_used}/{n_total} senders; "
          f"{n_neg} with MCC < 0; median={median_mcc:.3f}, mean={mean_mcc:.3f}")

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bins = np.linspace(-1.0, 1.0, 41)
    ax.hist(mccs, bins=bins, color="#4A4E69", edgecolor="white", alpha=0.88)

    # Chance line (per caption: "dashed line marks MCC = 0")
    ax.axvline(0.0, color="black", lw=1.2, ls="--", label="MCC = 0 (chance)")
    # Median marker
    ax.axvline(median_mcc, color="#C73E1D", lw=1.5, ls="-",
               label=rf"median = {median_mcc:.3f}")

    ax.set_xlabel("Per-sender MCC")
    ax.set_ylabel("Number of senders")
    ax.set_title(
        f"Per-sender MCC on the test partition\n"
        f"({n_used} senders with both benign and malicious test messages; "
        f"{n_neg} with MCC $< 0$)"
    )
    ax.set_xlim(-1.05, 1.05)
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig7] wrote {out_path}")


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
    ap.add_argument("--out", default="figs", help="Output directory")
    ap.add_argument("--threshold", type=float, default=0.1926,
                    help="Validation-selected decision threshold")
    a = ap.parse_args()

    if not os.path.exists(a.npz):
        raise SystemExit(f"NPZ not found: {a.npz}")

    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.npz, allow_pickle=False)
    y = d["y_test"].astype(np.int64)
    p = d["test_prob"].astype(np.float64)
    sender_id = d["sender_id"].astype(str) if "sender_id" in d.files else None

    print(f"[npz] loaded {len(y):,} test messages from {a.npz}")

    # Fig. 4: threshold sweep
    fig4_threshold_sweep(
        y, p, a.threshold,
        os.path.join(a.out, "fig4_threshold_sweep.pdf"),
    )

    # Fig. 7: per-sender MCC
    if sender_id is None:
        print("[fig7] sender_id missing from NPZ; figure skipped")
    else:
        fig7_per_sender_mcc(
            sender_id, y, p, a.threshold,
            os.path.join(a.out, "fig7_per_sender_mcc.pdf"),
        )


if __name__ == "__main__":
    main()