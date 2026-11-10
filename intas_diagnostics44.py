
"""
InTAS diagnostics and Uniform-DP evaluation.

Part 1:
    Spatial band cross-validation on Train.
    Train -> Validation diagnostic evaluation.
    Test labels are NOT used.

Part 2:
    Evaluation of previously generated Uniform-DP predictions.
    Requires saved test labels and predicted probabilities.
    Does NOT train TopTrust or modify the official split.

Examples:

    # Spatial feature diagnostics only
    python intas_diagnostics4.py --bands 5 --buffer 100

    # Evaluate saved Uniform-DP predictions
    python intas_diagnostics4.py --evaluate-npz uniform_dp_predictions.npz

Expected NPZ keys:
    y_test
    test_prob

Optional:
    y_val
    val_prob

If validation predictions are supplied, the threshold is selected
on Validation using the maximum F1 criterion.
Otherwise, the previously selected validation threshold is used.
"""

import argparse
import csv
import os

import numpy as np

from intas_diagnostics import (
    ALL_NAMES,
    GNN_NAMES,
    records_to_matrix,
)
from intas_diagnostics2 import edge_column


# ============================================================
# Feature configurations
# ============================================================

REL = [
    "dx",
    "dy",
    "dist_xy",
    "d_spd",
    "d_acl",
    "hed_diff_cos",
    "hed_diff_sin",
    "delay_s",
]

KIN = [
    "s_spd",
    "s_acl",
    "s_sin_hed",
    "s_cos_hed",
    "r_spd",
    "r_acl",
    "r_sin_hed",
    "r_cos_hed",
]

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


# ============================================================
# Data loading
# ============================================================

def load_records(records, split_name):
    """
    Convert InTAS records to a feature matrix.

    The edge feature is appended after validating row alignment.
    """

    X, y, _, bad = records_to_matrix(records, split_name)

    if bad or len(y) != len(records):
        raise RuntimeError(
            f"{split_name}: invalid records were skipped. "
            "Cannot safely align edge features."
        )

    edge = np.asarray(edge_column(records)).reshape(-1, 1)

    if len(edge) != len(X):
        raise RuntimeError(
            f"{split_name}: edge feature length mismatch."
        )

    X = np.column_stack([X, edge])
    y = np.asarray(y).astype(int).ravel()

    if not np.isfinite(X).all():
        raise ValueError(
            f"{split_name}: feature matrix contains NaN or Inf."
        )

    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            f"{split_name}: labels must be binary."
        )

    return X, y


# ============================================================
# Diagnostic classifier
# ============================================================

def fit_auc(X_train, y_train, X_eval, y_eval, columns):
    """
    Train a diagnostic HistGradientBoosting classifier
    and return ROC-AUC on the evaluation partition.
    """

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_train)) < 2:
        return float("nan")

    if len(np.unique(y_eval)) < 2:
        return float("nan")

    model = HistGradientBoostingClassifier(
        random_state=0
    )

    model.fit(
        X_train[:, columns],
        y_train
    )

    probabilities = model.predict_proba(
        X_eval[:, columns]
    )[:, 1]

    return roc_auc_score(
        y_eval,
        probabilities
    )


# ============================================================
# Spatial band cross-validation
# ============================================================

def run_spatial_diagnostics(bands=5, buffer=100.0):

    from spliteDataset import load_dataset

    if bands < 2:
        raise ValueError("--bands must be at least 2.")

    if buffer < 0:
        raise ValueError("--buffer cannot be negative.")

    train_records, val_records, _ = load_dataset()

    X_train, y_train = load_records(
        train_records, "Train"
    )

    X_val, y_val = load_records(
        val_records, "Validation"
    )

    feature_index = {
        name: i for i, name in enumerate(NAMES)
    }

    missing = sorted({
        feature
        for feature_list in SETS.values()
        for feature in feature_list
        if feature not in feature_index
    })

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    if "r_y" not in feature_index:
        raise ValueError(
            "Receiver y coordinate r_y is unavailable."
        )

    receiver_y = X_train[:, feature_index["r_y"]]

    edges = np.quantile(
        receiver_y,
        np.linspace(0, 1, bands + 1)
    )

    edges[-1] += 1e-6

    print("\n" + "=" * 88)
    print(
        f"SPATIAL BAND CV ON TRAIN "
        f"(bands={bands}, buffer={buffer:.1f} m)"
    )
    print("=" * 88)

    for b in range(bands):

        held = (
            (receiver_y >= edges[b]) &
            (receiver_y < edges[b + 1])
        )

        share = (
            y_train[held].mean()
            if held.sum() > 0
            else float("nan")
        )

        print(
            f"Band {b}: "
            f"y=[{edges[b]:.1f}, {edges[b + 1]:.1f}) "
            f"rows={held.sum():,} "
            f"malicious_share={share:.4f}"
        )

    results = []

    header = (
        f"\n{'feature set':<20}"
        + "".join(
            f"{('band' + str(b)):>9}"
            for b in range(bands)
        )
        + f"{'mean':>9}"
        + f"{'min':>9}"
        + f"{'Train->Val':>12}"
    )

    print(header)
    print("-" * len(header))

    for name, feature_names in SETS.items():

        columns = [
            feature_index[f]
            for f in feature_names
        ]

        aucs = []

        for b in range(bands):

            low = edges[b]
            high = edges[b + 1]

            held = (
                (receiver_y >= low) &
                (receiver_y < high)
            )

            keep = (
                (receiver_y < low - buffer) |
                (receiver_y >= high + buffer)
            )

            auc = fit_auc(
                X_train[keep],
                y_train[keep],
                X_train[held],
                y_train[held],
                columns
            )

            aucs.append(auc)

        val_auc = fit_auc(
            X_train,
            y_train,
            X_val,
            y_val,
            columns
        )

        mean_auc = (
            float(np.nanmean(aucs))
            if np.isfinite(aucs).any()
            else float("nan")
        )

        min_auc = (
            float(np.nanmin(aucs))
            if np.isfinite(aucs).any()
            else float("nan")
        )

        print(
            f"{name:<20}"
            + "".join(f"{v:>9.4f}" for v in aucs)
            + f"{mean_auc:>9.4f}"
            + f"{min_auc:>9.4f}"
            + f"{val_auc:>12.4f}"
        )

        row = {
            "feature_set": name,
            "mean_auc": mean_auc,
            "min_auc": min_auc,
            "train_val_auc": val_auc,
        }

        for b, auc in enumerate(aucs):
            row[f"band{b}"] = auc

        results.append(row)

    output_file = "spatial_robustness_results.csv"

    if results:
        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(results[0].keys())
            )

            writer.writeheader()
            writer.writerows(results)

    print(f"\nResults saved to: {output_file}")

    print(
        "\nNote: This is a feature-level diagnostic experiment, "
        "not an end-to-end TopTrust robustness evaluation."
    )


# ============================================================
# Validation threshold selection
# ============================================================

def select_validation_threshold(y_val, val_prob):

    from sklearn.metrics import f1_score

    y_val = np.asarray(y_val).astype(int).ravel()
    val_prob = np.asarray(val_prob).astype(float).ravel()

    if len(y_val) != len(val_prob):
        raise ValueError(
            "Validation labels and probabilities have different lengths."
        )

    if not np.isfinite(val_prob).all():
        raise ValueError(
            "Validation probabilities contain NaN or Inf."
        )

    if len(np.unique(y_val)) < 2:
        raise ValueError(
            "Validation must contain both classes."
        )

    # Evaluate candidate thresholds using validation only.
    thresholds = np.unique(
        np.concatenate([
            np.linspace(0.001, 0.999, 999),
            val_prob
        ])
    )

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:

        predictions = (
            val_prob >= threshold
        ).astype(int)

        score = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


# ============================================================
# Uniform-DP test evaluation
# ============================================================

def evaluate_uniform_dp(
    y_test,
    test_prob,
    threshold=0.133019,
    seed=42
):

    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
        classification_report,
    )

    y_test = np.asarray(y_test).astype(int).ravel()
    test_prob = np.asarray(test_prob).astype(float).ravel()

    if len(y_test) != len(test_prob):
        raise ValueError(
            "Test labels and probabilities have different lengths."
        )

    if len(y_test) == 0:
        raise ValueError("Test data is empty.")

    if not np.isfinite(test_prob).all():
        raise ValueError(
            "Test probabilities contain NaN or Inf."
        )

    if np.any((test_prob < 0) | (test_prob > 1)):
        raise ValueError(
            "Predicted probabilities must be between 0 and 1."
        )

    if not np.isin(y_test, [0, 1]).all():
        raise ValueError(
            "Test labels must be binary."
        )

    if len(np.unique(y_test)) < 2:
        raise ValueError(
            "Test partition must contain both classes."
        )

    y_pred = (
        test_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test, y_pred
    )

    roc_auc = roc_auc_score(
        y_test, test_prob
    )

    pr_auc = average_precision_score(
        y_test, test_prob
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    results = {
        "seed": seed,
        "threshold": threshold,
        "test_size": len(y_test),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    print("\n" + "=" * 70)
    print("UNIFORM-DP TEST EVALUATION")
    print("=" * 70)

    print(f"Seed:       {seed}")
    print(f"Threshold:  {threshold:.6f}")
    print(f"Test size:  {len(y_test):,}")

    print("\nDetection Metrics")
    print("-" * 70)

    print(f"Precision:  {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:     {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-score:   {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"MCC:        {mcc:.4f}")
    print(f"ROC-AUC:    {roc_auc:.4f}")
    print(f"PR-AUC:     {pr_auc:.4f}")

    print("\nConfusion Matrix")
    print("-" * 70)

    print("                 Predicted")
    print("                 Benign   Malicious")
    print(f"Actual Benign    {tn:7,}   {fp:9,}")
    print(f"Actual Malicious {fn:7,}   {tp:9,}")

    print("\nClassification Report")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Benign", "Malicious"],
            digits=4,
            zero_division=0
        )
    )

    output_file = "uniform_dp_test_results.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(results.keys())
        )

        writer.writeheader()
        writer.writerow(results)

    print(f"Results saved to: {output_file}")

    return results


# ============================================================
# Load saved predictions and evaluate
# ============================================================

def evaluate_from_npz(path, fixed_threshold, seed):

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    data = np.load(path)

    required = {"y_test", "test_prob"}

    missing = required - set(data.files)

    if missing:
        raise ValueError(
            f"NPZ is missing required arrays: {sorted(missing)}"
        )

    y_test = data["y_test"]
    test_prob = data["test_prob"]

    if "y_val" in data.files and "val_prob" in data.files:

        threshold, val_f1 = select_validation_threshold(
            data["y_val"],
            data["val_prob"]
        )

        print(
            f"\nValidation-selected threshold: {threshold:.6f}"
        )

        print(
            f"Validation F1 at selected threshold: {val_f1:.4f}"
        )

    else:

        threshold = fixed_threshold

        print(
            "\nUsing previously selected validation threshold: "
            f"{threshold:.6f}"
        )

    return evaluate_uniform_dp(
        y_test=y_test,
        test_prob=test_prob,
        threshold=threshold,
        seed=seed
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="InTAS spatial diagnostics and Uniform-DP evaluation"
    )

    parser.add_argument(
        "--bands",
        type=int,
        default=5
    )

    parser.add_argument(
        "--buffer",
        type=float,
        default=100.0
    )

    parser.add_argument(
        "--evaluate-npz",
        type=str,
        default=None,
        help="NPZ containing y_test and test_prob"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.133019,
        help="Previously selected validation threshold"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--skip-spatial",
        action="store_true",
        help="Skip spatial diagnostics"
    )

    args = parser.parse_args()

    if args.evaluate_npz is not None:

        evaluate_from_npz(
            args.evaluate_npz,
            args.threshold,
            args.seed
        )

    elif not args.skip_spatial:

        run_spatial_diagnostics(
            bands=args.bands,
            buffer=args.buffer
        )

    else:

        parser.error(
            "Specify --evaluate-npz or enable spatial diagnostics."
        )


if __name__ == "__main__":
    main()
