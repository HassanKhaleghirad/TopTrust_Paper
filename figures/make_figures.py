"""
make_figures.py

Generate all figures for the TOPTrust paper that can be derived
from the available experimental artifacts:

  * uniform_dp_predictions_eps8.0_seed42_rounds150_cmax1.0_edge_rel_kin.npz
  * the training log (stdout), saved separately as train_log.txt

Figures that require experiments which have not been run yet
(privacy-utility curve at other epsilons, gradient-perturbation
robustness, cold-start with bootstrap, latency measurements)
are explicitly reported as unavailable and are NOT drawn with
fabricated data.

Usage:
    python make_figures.py \
        --npz uniform_dp_predictions_eps8.0_seed42_rounds150_cmax1.0_edge_rel_kin.npz \
        --log train_log.txt \
        --out figs/

Outputs (into --out):
    fig2_score_distribution.pdf   per-class score histogram (Val + Test)
    fig3_pr_roc_curves.pdf        PR and ROC curves (Val + Test)
    fig4_confusion_matrices.pdf   confusion matrices (Val + Test)
    fig5_convergence.pdf          validation metrics over rounds
    fig6_per_sender_mcc.pdf       per-sender MCC distribution
    fig7_threshold_sweep.pdf      F1 / MCC / precision / recall vs threshold
    fig8_epsilon_budget.pdf       epsilon spent vs round
    fig_summary.csv               all metrics in a single CSV
"""

import argparse
import csv
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
})

C_BENIGN = "#2E86AB"
C_MAL    = "#C73E1D"
C_NEUTRAL = "#6C757D"


# -----------------------------------------------------------------------------
# Metric helpers (kept in sync with the main script)
# -----------------------------------------------------------------------------
def metrics_from_probs(probs, labels, threshold=0.5):
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    pred = (probs >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-12))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=prec, recall=rec, f1=f1, mcc=mcc)


def roc_curve_np(probs, labels):
    """Return FPR, TPR arrays for the ROC curve."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    order = np.argsort(-probs)
    probs_sorted = probs[order]
    labels_sorted = labels[order]
    distinct = np.ones(len(probs_sorted), dtype=bool)
    if len(probs_sorted) > 1:
        distinct[:-1] = probs_sorted[:-1] != probs_sorted[1:]
    cum_tp = np.cumsum(labels_sorted)[distinct]
    cum_fp = np.cumsum(1 - labels_sorted)[distinct]
    npos = max(int(labels.sum()), 1)
    nneg = max(int(len(labels) - labels.sum()), 1)
    tpr = cum_tp / npos
    fpr = cum_fp / nneg
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return fpr, tpr


def pr_curve_np(probs, labels):
    """Return recall, precision arrays for the PR curve."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    order = np.argsort(-probs)
    probs_sorted = probs[order]
    labels_sorted = labels[order]
    distinct = np.ones(len(probs_sorted), dtype=bool)
    if len(probs_sorted) > 1:
        distinct[:-1] = probs_sorted[:-1] != probs_sorted[1:]
    cum_tp = np.cumsum(labels_sorted)[distinct]
    cum_fp = np.cumsum(1 - labels_sorted)[distinct]
    npos = max(int(labels.sum()), 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    recall = cum_tp / npos
    return recall, precision


def auc_from_curve(x, y):
    """Trapezoidal AUC."""
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def ap_from_curve(recall, precision):
    """Average precision as the integral of the PR curve."""
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


# -----------------------------------------------------------------------------
# Figure 2: score distribution
# -----------------------------------------------------------------------------
def fig_score_distribution(y, p, out_path, split_name):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bins = np.linspace(0.0, 1.0, 60)
    ax.hist(p[y == 0], bins=bins, alpha=0.75, color=C_BENIGN, label="Benign", edgecolor="none")
    ax.hist(p[y == 1], bins=bins, alpha=0.75, color=C_MAL, label="Malicious", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted probability of malicious")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"{split_name}: score distribution (Uniform-DP, ε=8)")
    ax.legend(loc="upper center", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 3: ROC and PR curves
# -----------------------------------------------------------------------------
def fig_roc_pr(y, p, out_path, split_name):
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4))

    fpr, tpr = roc_curve_np(p, y)
    auc = auc_from_curve(fpr, tpr)
    ax_roc.plot(fpr, tpr, color=C_MAL, lw=2, label=f"AUC = {auc:.4f}")
    ax_roc.plot([0, 1], [0, 1], "--", color=C_NEUTRAL, lw=1)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title(f"{split_name}: ROC curve")
    ax_roc.legend(loc="lower right", frameon=False)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)

    rec, prec = pr_curve_np(p, y)
    ap = ap_from_curve(rec, prec)
    ax_pr.plot(rec, prec, color=C_MAL, lw=2, label=f"AP = {ap:.4f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"{split_name}: precision-recall curve")
    ax_pr.legend(loc="lower left", frameon=False)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.02)

    fig.savefig(out_path)
    plt.close(fig)
    return dict(roc_auc=auc, pr_auc=ap)


# -----------------------------------------------------------------------------
# Figure 4: confusion matrices
# -----------------------------------------------------------------------------
def fig_confusion(y, p, threshold, out_path, split_name):
    m = metrics_from_probs(p, y, threshold)
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for (i, j), v in np.ndenumerate(cm):
        color = "white" if v > cm.max() / 2 else "black"
        ax.text(j, i, f"{v:,}", ha="center", va="center", color=color, fontsize=12)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred. benign", "Pred. malicious"])
    ax.set_yticklabels(["Benign", "Malicious"])
    ax.set_xlabel("Prediction"); ax.set_ylabel("Ground truth")
    ax.set_title(f"{split_name}: confusion matrix (τ={threshold:.4f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path)
    plt.close(fig)
    return m


# -----------------------------------------------------------------------------
# Figure 5: convergence curve from log
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
    rounds, f1s, mccs, aucs, aps, lams, epss = [], [], [], [], [], [], []
    if not os.path.exists(path):
        return None
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
            lams.append(float(m.group("lam")))
            epss.append(float(m.group("eps")))
    if not rounds:
        return None
    return dict(rounds=np.array(rounds),
                f1=np.array(f1s), mcc=np.array(mccs),
                auc=np.array(aucs), ap=np.array(aps),
                lam=np.array(lams), eps=np.array(epss))


def fig_convergence(log, out_path):
    if log is None:
        print("[SKIP] fig5_convergence: log file not found or unparsable")
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    ax1.plot(log["rounds"], log["f1"], "-o", ms=3, color="#1F77B4", label="F1")
    ax1.plot(log["rounds"], log["mcc"], "-s", ms=3, color="#FF7F0E", label="MCC")
    ax1.plot(log["rounds"], log["auc"], "-^", ms=3, color="#2CA02C", label="ROC-AUC")
    ax1.plot(log["rounds"], log["ap"], "-d", ms=3, color="#D62728", label="PR-AUC")
    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("Validation score (threshold 0.5)")
    ax1.set_title("(a) Convergence")
    ax1.legend(loc="lower right", frameon=False, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.02, 1.02)

    ax2.plot(log["rounds"], log["eps"], "-o", ms=3, color="#9467BD")
    ax2.set_xlabel("Communication round")
    ax2.set_ylabel("Epsilon spent (RDP accountant)")
    ax2.set_title("(b) Privacy budget consumption")
    ax2.grid(True, alpha=0.3)

    fig.savefig(out_path)
    plt.close(fig)
    return log


# -----------------------------------------------------------------------------
# Figure 6: per-sender MCC distribution
# -----------------------------------------------------------------------------
def per_sender_metrics(sender_id, y, p, threshold):
    by_sender = defaultdict(lambda: ([], []))
    for s, yi, pi in zip(sender_id, y, p):
        by_sender[s][0].append(int(yi))
        by_sender[s][1].append(float(pi))
    out = {}
    for s, (yy, pp) in by_sender.items():
        yy = np.asarray(yy); pp = np.asarray(pp)
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        out[s] = metrics_from_probs(pp, yy, threshold)
    return out


def fig_per_sender_mcc(sender_id, y, p, threshold, out_path):
    per = per_sender_metrics(sender_id, y, p, threshold)
    if not per:
        print("[SKIP] fig6_per_sender_mcc: no sender has both classes")
        return None
    mccs = np.array([m["mcc"] for m in per.values()])
    n_per = len(mccs)
    neg = int((mccs < 0).sum())

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(mccs, bins=np.linspace(-1, 1, 41), color=C_NEUTRAL, edgecolor="white")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Per-sender MCC")
    ax.set_ylabel("Number of senders")
    ax.set_title(f"Per-sender MCC distribution (n={n_per} senders; {neg} with MCC < 0)")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)

    worst = sorted(per.items(), key=lambda kv: kv[1]["mcc"])[:10]
    print(f"[per-sender] {neg}/{n_per} senders have MCC < 0")
    for s, m in worst:
        print(f"  {s:>12s}  MCC={m['mcc']:+.3f}  "
              f"F1={m['f1']:.3f}  n={m['tp']+m['fp']+m['tn']+m['fn']}")
    return per


# -----------------------------------------------------------------------------
# Figure 7: threshold sweep
# -----------------------------------------------------------------------------
def fig_threshold_sweep(y, p, out_path, split_name):
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s, mccs, precs, recs = [], [], [], []
    for t in thresholds:
        m = metrics_from_probs(p, y, t)
        f1s.append(m["f1"]); mccs.append(m["mcc"])
        precs.append(m["precision"]); recs.append(m["recall"])

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(thresholds, f1s, lw=2, color="#1F77B4", label="F1")
    ax.plot(thresholds, mccs, lw=2, color="#FF7F0E", label="MCC")
    ax.plot(thresholds, precs, lw=1.5, ls="--", color="#2CA02C", label="Precision")
    ax.plot(thresholds, recs, lw=1.5, ls="--", color="#D62728", label="Recall")
    best = int(np.argmax(f1s))
    ax.axvline(thresholds[best], color="black", lw=1, ls=":")
    ax.text(thresholds[best], 0.05, f" τ*={thresholds[best]:.3f}",
            fontsize=8, rotation=90, va="bottom")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"{split_name}: threshold sweep")
    ax.legend(loc="center right", frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.02)
    fig.savefig(out_path)
    plt.close(fig)
    return thresholds[best], f1s[best]


# -----------------------------------------------------------------------------
# Figure 8: privacy budget
# -----------------------------------------------------------------------------
def fig_epsilon_budget(log, out_path):
    if log is None:
        print("[SKIP] fig8_epsilon_budget: log file not found")
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(log["rounds"], log["eps"], "-o", ms=3, color="#9467BD", lw=2)
    ax.axhline(log["eps"][-1], color="black", lw=1, ls="--",
               label=f"Final ε = {log['eps'][-1]:.2f}")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Epsilon spent")
    ax.set_title("RDP-composed privacy budget over rounds")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Summary CSV
# -----------------------------------------------------------------------------
def write_summary_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "n", "tp", "fp", "tn", "fn",
                                               "precision", "recall", "f1", "mcc",
                                               "roc_auc", "pr_auc", "threshold"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[CSV] wrote {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True,
                    help="NPZ produced by Edited_toptrust_version4.py (--mode uniform)")
    ap.add_argument("--log", default="train_log.txt",
                    help="Training stdout log (must contain per-round lines)")
    ap.add_argument("--out", default="figs", help="Output directory")
    ap.add_argument("--threshold", type=float, default=0.1926,
                    help="Validation-selected threshold (used for confusion matrices)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.npz, allow_pickle=False)
    y = d["y_test"].astype(np.int64)
    p = d["test_prob"].astype(np.float64)
    sender_id = d["sender_id"].astype(str) if "sender_id" in d.files else None
    train_senders = set(d["train_sender_ids"].astype(str)) if "train_sender_ids" in d.files else set()

    print(f"[npz] loaded {len(y):,} messages from {a.npz}")
    print(f"[npz] malicious={int(y.sum()):,}  benign={int((1 - y).sum()):,}")
    if sender_id is not None:
        print(f"[npz] unique test senders: {len(set(sender_id)):,}")

    # Unseen-sender split
    if sender_id is not None and train_senders:
        unseen_mask = np.array([s not in train_senders for s in sender_id])
        print(f"[npz] unseen senders: {int(unseen_mask.sum()):,} messages")
    else:
        unseen_mask = None

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    test_metrics = metrics_from_probs(p, y, a.threshold)
    fpr, tpr = roc_curve_np(p, y)
    test_roc_auc = auc_from_curve(fpr, tpr)
    rec, prec = pr_curve_np(p, y)
    test_pr_auc = ap_from_curve(rec, prec)

    print("\n=== Test (full) ===")
    print(f"  F1={test_metrics['f1']:.4f}  MCC={test_metrics['mcc']:.4f}  "
          f"Prec={test_metrics['precision']:.4f}  Rec={test_metrics['recall']:.4f}  "
          f"ROC-AUC={test_roc_auc:.4f}  PR-AUC={test_pr_auc:.4f}")

    rows = [dict(split="test_full", n=len(y),
                 tp=test_metrics["tp"], fp=test_metrics["fp"],
                 tn=test_metrics["tn"], fn=test_metrics["fn"],
                 precision=test_metrics["precision"], recall=test_metrics["recall"],
                 f1=test_metrics["f1"], mcc=test_metrics["mcc"],
                 roc_auc=test_roc_auc, pr_auc=test_pr_auc,
                 threshold=a.threshold)]

    if unseen_mask is not None and unseen_mask.sum() > 0:
        y_u, p_u = y[unseen_mask], p[unseen_mask]
        m_u = metrics_from_probs(p_u, y_u, a.threshold)
        fpr_u, tpr_u = roc_curve_np(p_u, y_u)
        roc_u = auc_from_curve(fpr_u, tpr_u)
        rec_u, prec_u = pr_curve_np(p_u, y_u)
        pr_u = ap_from_curve(rec_u, prec_u)
        print("\n=== Test (unseen senders only) ===")
        print(f"  F1={m_u['f1']:.4f}  MCC={m_u['mcc']:.4f}  "
              f"Prec={m_u['precision']:.4f}  Rec={m_u['recall']:.4f}  "
              f"ROC-AUC={roc_u:.4f}  PR-AUC={pr_u:.4f}")
        rows.append(dict(split="test_unseen", n=int(unseen_mask.sum()),
                         tp=m_u["tp"], fp=m_u["fp"], tn=m_u["tn"], fn=m_u["fn"],
                         precision=m_u["precision"], recall=m_u["recall"],
                         f1=m_u["f1"], mcc=m_u["mcc"],
                         roc_auc=roc_u, pr_auc=pr_u, threshold=a.threshold))

    write_summary_csv(rows, os.path.join(a.out, "fig_summary.csv"))

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    fig_score_distribution(y, p, os.path.join(a.out, "fig2_score_distribution.pdf"),
                           "Test")
    fig_roc_pr(y, p, os.path.join(a.out, "fig3_pr_roc_curves.pdf"), "Test")
    fig_confusion(y, p, a.threshold, os.path.join(a.out, "fig4_confusion_matrices.pdf"),
                  "Test")
    fig_threshold_sweep(y, p, os.path.join(a.out, "fig7_threshold_sweep.pdf"), "Test")

    log = parse_log(a.log)
    if log is None:
        print(f"\n[WARN] Could not parse {a.log}. Convergence and budget figures skipped.")
        print("       Save the full stdout of the run to --log to enable fig5 and fig8.")
    else:
        fig_convergence(log, os.path.join(a.out, "fig5_convergence.pdf"))
        fig_epsilon_budget(log, os.path.join(a.out, "fig8_epsilon_budget.pdf"))
        print(f"[log] parsed {len(log['rounds'])} round entries from {a.log}")

    if sender_id is not None:
        fig_per_sender_mcc(sender_id, y, p, a.threshold,
                           os.path.join(a.out, "fig6_per_sender_mcc.pdf"))

    # ------------------------------------------------------------------
    # Unavailable figures
    # ------------------------------------------------------------------
    print("\n=== Figures NOT generated (require experiments not yet run) ===")
    unavailable = [
        ("privacy-utility curve at ε=0.5, 1, 2, 5",
         "run: --epsilon {0.5,1,2,5} with the same configuration"),
        ("gradient-perturbation robustness",
         "run: injected gradient noise experiment (not implemented yet)"),
        ("cold-start with bootstrap enabled",
         "run: with --enable-bootstrap and a held-out-unseen protocol"),
        ("latency vs number of vehicles",
         "run: timing harness over vehicle-count sweeps"),
        ("shuffled-degree / mean-matched DP ablations",
         "run: implement the corresponding clipping modes"),
    ]
    for name, need in unavailable:
        print(f"  - {name}")
        print(f"      needed: {need}")

    print(f"\nFigures written to {a.out}/")


if __name__ == "__main__":
    main()