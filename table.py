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