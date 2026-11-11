"""
make_fig7_per_sender_rich.py

Rich two-panel figure for per-sender MCC on the test partition.

Panel (a): distribution of per-sender MCC, with median, mean, and
           reference lines at MCC=0 and MCC=+1. Annotates key statistics.
Panel (b): per-sender MCC vs number of test messages per sender (scatter),
           colored by the sender's fraction of malicious messages.
           Reveals that senders with low MCC typically also have few
           test messages (data-limited regime) and/or a skewed class ratio.

Only senders whose test records contain BOTH benign and malicious messages
are included (otherwise MCC is undefined).

Usage:
    python make_fig7_per_sender_rich.py \
        --npz uniform_dp_predictions_eps8.0_seed42_rounds150_cmax1.0_edge_rel_kin.npz \
        --out figs/fig7_per_sender_mcc.pdf \
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
# Metric helper
# -----------------------------------------------------------------------------
def mcc_one(y, p, threshold):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-12))
    return (tp * tn - fp * fn) / den if den > 0 else 0.0


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_per_sender(npz_path, threshold):
    d = np.load(npz_path, allow_pickle=False)
    if "sender_id" not in d.files:
        raise SystemExit(
            f"NPZ has no 'sender_id' field: {npz_path}\n"
            f"Available fields: {d.files}\n"
            "Regenerate the NPZ with the updated main() that stores sender_id."
        )
    y = d["y_test"].astype(np.int64)
    p = d["test_prob"].astype(np.float64)
    sid = d["sender_id"].astype(str)

    by_sender = defaultdict(lambda: ([], []))
    for s, yi, pi in zip(sid, y, p):
        by_sender[s][0].append(int(yi))
        by_sender[s][1].append(float(pi))

    rows = []
    for s, (yy, pp) in by_sender.items():
        yy = np.asarray(yy)
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        pp = np.asarray(pp)
        m = mcc_one(yy, pp, threshold)
        rows.append(dict(
            sender=s,
            n=int(len(yy)),
            n_mal=int(yy.sum()),
            frac_mal=float(yy.mean()),
            mcc=float(m),
        ))
    return rows


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
def make_figure(rows, out_path, threshold):
    if not rows:
        raise SystemExit("No sender has both classes in the test partition.")

    mccs = np.array([r["mcc"] for r in rows])
    sizes = np.array([r["n"] for r in rows])
    frac_mal = np.array([r["frac_mal"] for r in rows])

    n = len(mccs)
    n_neg = int((mccs < 0).sum())
    n_strong = int((mccs > 0.5).sum())
    median_mcc = float(np.median(mccs))
    mean_mcc = float(np.mean(mccs))
    q25, q75 = np.percentile(mccs, [25, 75])

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(12.5, 4.2),
        gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.30},
    )

    # -----------------------------------------------------------------
    # Panel (a): histogram of per-sender MCC
    # -----------------------------------------------------------------
    bins = np.linspace(-1.0, 1.0, 41)
    ax_a.hist(mccs, bins=bins, color="#4A4E69", edgecolor="white", alpha=0.88)
    ax_a.axvline(0.0, color="black", lw=1.2, ls="--", label="MCC = 0 (chance)")
    ax_a.axvline(median_mcc, color="#C73E1D", lw=1.6,
                 label=rf"median = {median_mcc:.3f}")
    ax_a.axvspan(q25, q75, color="#C73E1D", alpha=0.06,
                 label=rf"IQR = [{q25:.3f}, {q75:.3f}]")

    # Statistics box
    stats_text = (
        rf"$n$ senders = {n}" "\n"
        rf"median MCC = {median_mcc:.3f}" "\n"
        rf"mean MCC   = {mean_mcc:.3f}" "\n"
        rf"MCC $>$ 0.5: {n_strong} ({100*n_strong/n:.1f}\%)" "\n"
        rf"MCC $<$ 0:  {n_neg} ({100*n_neg/n:.1f}\%)"
    )
    ax_a.text(
        0.03, 0.97, stats_text,
        transform=ax_a.transAxes,
        fontsize=8.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor="white", edgecolor="#4A4E69",
                  linewidth=0.8, alpha=0.95),
    )

    ax_a.set_xlabel("Per-sender MCC")
    ax_a.set_ylabel("Number of senders")
    ax_a.set_title("(a) Per-sender MCC distribution")
    ax_a.set_xlim(-1.05, 1.05)
    ax_a.legend(loc="lower left", frameon=False, fontsize=8.0)

    # -----------------------------------------------------------------
    # Panel (b): MCC vs n_messages, colored by fraction malicious
    # -----------------------------------------------------------------
    # Split by sign for visual emphasis
    neg = mccs < 0
    sc = ax_b.scatter(
        sizes[~neg], mccs[~neg],
        c=frac_mal[~neg], cmap="viridis", vmin=0, vmax=1,
        s=28, alpha=0.75, edgecolor="white", linewidth=0.5,
        label=r"MCC $\geq 0$",
    )
    ax_b.scatter(
        sizes[neg], mccs[neg],
        c=frac_mal[neg], cmap="viridis", vmin=0, vmax=1,
        s=44, alpha=0.95, edgecolor="#C73E1D", linewidth=1.0,
        marker="X", label=r"MCC $< 0$",
    )

    ax_b.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.8)
    ax_b.axhline(median_mcc, color="#C73E1D", lw=1.0, ls=":",
                 alpha=0.7, label=f"median = {median_mcc:.3f}")
    ax_b.set_xscale("log")
    ax_b.set_xlabel("Number of test messages per sender (log scale)")
    ax_b.set_ylabel("Per-sender MCC")
    ax_b.set_title("(b) MCC vs. per-sender sample count")
    ax_b.set_ylim(-1.05, 1.05)
    ax_b.legend(loc="lower right", frameon=False, fontsize=8.0)
    ax_b.xaxis.set_major_locator(MaxNLocator(nbins=6))

    cbar = fig.colorbar(sc, ax=ax_b, pad=0.02, fraction=0.045)
    cbar.set_label("Fraction malicious in test", fontsize=8.5)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig7] wrote {out_path}")
    print(f"       n={n} senders, median={median_mcc:.3f}, "
          f"mean={mean_mcc:.3f}, {n_neg} with MCC<0")


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
    ap.add_argument("--out", default="figs/fig7_per_sender_mcc.pdf")
    ap.add_argument("--threshold", type=float, default=0.1926)
    a = ap.parse_args()

    if not os.path.exists(a.npz):
        raise SystemExit(f"NPZ not found: {a.npz}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows = load_per_sender(a.npz, a.threshold)
    make_figure(rows, a.out, a.threshold)


if __name__ == "__main__":
    main()