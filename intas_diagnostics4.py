"""
InTAS diagnostics, part 4  (read-only: no TopTrust training, DP, or split changes).

Test positions lie ~1.8 km away from anything seen in Train, so absolute coordinates cannot transfer.
This script chooses input features WITHOUT touching Test labels, by simulating that shift inside Train:

  * Train is cut into 5 contiguous bands along the receiver's y coordinate (receiver position is honest).
  * Leave-one-band-out: train on the other bands (rows within --buffer metres of the held-out band are
    dropped), evaluate on the held-out band.
  * Also reports the ordinary Train -> Validation score for the same feature sets.

Feature sets compared (HistGradientBoosting):
  abs_pos          s_x, s_y
  edge             distance_to_road_edge (extra dataset field; CAUTION: confirm how it is computed)
  rel              relative sender-receiver features
  rel+kin          relative + speed/accel/heading of both vehicles       (no map, no position)
  edge+rel         edge + relative
  edge+rel+kin     edge + relative + kinematics                          (location-invariant)
  all14+edge       the 14 GNN inputs + edge                              (contains absolute positions)

Requires intas_diagnostics.py and intas_diagnostics2.py in the same folder. Test labels are not used.

Run:  python intas_diagnostics4.py
"""
import argparse

import numpy as np

from intas_diagnostics import ALL_NAMES, GNN_NAMES, records_to_matrix, roc_auc
from intas_diagnostics2 import edge_column

REL = ["dx", "dy", "dist_xy", "d_spd", "d_acl", "hed_diff_cos", "hed_diff_sin", "delay_s"]
KIN = ["s_spd", "s_acl", "s_sin_hed", "s_cos_hed", "r_spd", "r_acl", "r_sin_hed", "r_cos_hed"]
NAMES = ALL_NAMES + ["edge"]
SETS = {
    "abs_pos": ["s_x", "s_y"],
    "edge": ["edge"],
    "rel": REL,
    "rel+kin": REL + KIN,
    "edge+rel": ["edge"] + REL,
    "edge+rel+kin": ["edge"] + REL + KIN,
    "all14+edge": GNN_NAMES + ["edge"],
}


def fit_auc(Xa, ya, Xb, yb, cols):
    from sklearn.ensemble import HistGradientBoostingClassifier
    if len(set(yb.tolist())) < 2 or len(set(ya.tolist())) < 2:
        return float("nan")
    m = HistGradientBoostingClassifier(random_state=0).fit(Xa[:, cols], ya)
    return roc_auc(m.predict_proba(Xb[:, cols])[:, 1], yb)


def load(recs, name):
    X, y, s, bad = records_to_matrix(recs, name)
    if bad or len(y) != len(recs):
        raise SystemExit(f"{name}: records were skipped during parsing; alignment with edge column is unsafe")
    return np.column_stack([X, edge_column(recs)]), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=int, default=5)
    ap.add_argument("--buffer", type=float, default=100.0, help="metres dropped around the held-out band")
    a = ap.parse_args()

    from spliteDataset import load_dataset
    tr, va, _ = load_dataset()
    Xtr, ytr = load(tr, "Train")
    Xva, yva = load(va, "Validation")
    ix = {n: i for i, n in enumerate(NAMES)}
    ry = Xtr[:, ix["r_y"]]
    edges = np.quantile(ry, np.linspace(0, 1, a.bands + 1))
    edges[-1] += 1e-6

    print("=" * 88)
    print(f"SPATIAL BAND CV on Train (bands along receiver y, buffer={a.buffer:.0f} m)")
    print("=" * 88)
    for b in range(a.bands):
        m = (ry >= edges[b]) & (ry < edges[b + 1])
        print(f"band {b}: y in [{edges[b]:.0f}, {edges[b + 1]:.0f})  rows={m.sum():,}  malicious share={ytr[m].mean():.3f}")

    print(f"\n{'feature set':<16}" + "".join(f"{'band' + str(b):>9}" for b in range(a.bands))
          + f"{'mean':>9}{'min':>9}{'Train->Val':>12}")
    for name, cols_n in SETS.items():
        cols = [ix[c] for c in cols_n]
        aucs = []
        for b in range(a.bands):
            lo, hi = edges[b], edges[b + 1]
            held = (ry >= lo) & (ry < hi)
            keep = (ry < lo - a.buffer) | (ry >= hi + a.buffer)
            aucs.append(fit_auc(Xtr[keep], ytr[keep], Xtr[held], ytr[held], cols))
        val = fit_auc(Xtr, ytr, Xva, yva, cols)
        print(f"{name:<16}" + "".join(f"{v:>9.4f}" for v in aucs)
              + f"{np.nanmean(aucs):>9.4f}{np.nanmin(aucs):>9.4f}{val:>12.4f}")
    print("\nRead: a feature set that stays high on EVERY band (high min) is the safer input for a spatially "
          "shifted Test. Bands at the edge of the y range are the closest analogue of Test.")

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


def evaluate_uniform_dp(
    y_test,
    y_prob,
    threshold=0.133019,
    seed=42
):
    """
    Evaluate the Uniform-DP model on the official test partition.

    Args:
        y_test: Ground-truth binary labels (0=benign, 1=malicious).
        y_prob: Predicted probabilities for the malicious class.
        threshold: Threshold selected using validation data only.
        seed: Random seed used in the experiment.
    """

    y_test = np.asarray(y_test).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()

    if len(y_test) != len(y_prob):
        raise ValueError("y_test and y_prob have different lengths.")

    if not np.isfinite(y_prob).all():
        raise ValueError("Predicted probabilities contain NaN or Inf.")

    if not np.isin(y_test, [0, 1]).all():
        raise ValueError("Labels must be binary: 0 or 1.")

    # Apply the validation-selected threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Detection metrics
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_prob)

    pr_auc = average_precision_score(y_test, y_prob)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    print("\n" + "=" * 65)
    print("UNIFORM-DP TEST EVALUATION")
    print("=" * 65)

    print(f"Seed:       {seed}")
    print(f"Threshold:  {threshold:.6f}")
    print(f"Test size:  {len(y_test):,}")

    print("\nDetection Metrics")
    print("-" * 65)

    print(f"Precision:  {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:     {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-score:   {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"MCC:        {mcc:.4f}")
    print(f"ROC-AUC:    {roc_auc:.4f}")
    print(f"PR-AUC:     {pr_auc:.4f}")

    print("\nConfusion Matrix")
    print("-" * 65)

    print(f"TN: {tn:,}")
    print(f"FP: {fp:,}")
    print(f"FN: {fn:,}")
    print(f"TP: {tp:,}")

    print("\nMatrix:")
    print("                 Predicted")
    print("              Benign  Malicious")
    print(f"Actual Benign  {tn:7,}  {fp:9,}")
    print(f"Actual Malicious {fn:7,}  {tp:9,}")

    print("\nClassification Report")
    print("-" * 65)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Malicious"],
            digits=4,
            zero_division=0
        )
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }
if __name__ == "__main__":
    main()