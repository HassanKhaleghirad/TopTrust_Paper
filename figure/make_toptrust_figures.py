
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

# Consistent, publication-friendly typography.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def save_figure(fig, stem):
    """Save a vector PDF and a high-resolution PNG."""
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------
# Figure 1: Official InTAS split class distribution
# ------------------------------------------------------------
splits = ["Train", "Validation", "Test"]
benign = np.array([29870, 14730, 35549])
malicious = np.array([7831, 3247, 8710])
totals = benign + malicious

fig, ax = plt.subplots(figsize=(6.2, 3.5))
x = np.arange(len(splits))

ax.bar(x, benign, label="Benign")
ax.bar(x, malicious, bottom=benign, label="Malicious")

for i, (b, m, total) in enumerate(zip(benign, malicious, totals)):
    ax.text(i, b / 2, f"{b:,}", ha="center", va="center", fontsize=8)
    ax.text(i, b + m / 2, f"{m:,}", ha="center", va="center", fontsize=8)
    ax.text(i, total + 500, f"N={total:,}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x, splits)
ax.set_ylabel("Number of messages")
ax.set_title("Class distribution in the official InTAS splits")
ax.set_ylim(0, max(totals) * 1.14)
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save_figure(fig, "fig1_intas_class_distribution")


# ------------------------------------------------------------
# Figure 2: Validation vs. Test performance
# ------------------------------------------------------------
metrics = ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "MCC"]
validation = np.array([0.8763, 0.7869, 0.7139,
                       0.8571, 0.6116, 0.6764])
test = np.array([0.7971, 0.6964, 0.6248,
                 0.8741, 0.4861, 0.5972])

x = np.arange(len(metrics))
width = 0.36

fig, ax = plt.subplots(figsize=(7.0, 3.7))
ax.bar(x - width / 2, validation, width, label="Validation")
ax.bar(x + width / 2, test, width, label="Test")

ax.set_xticks(x, metrics)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.0)
ax.set_title("Validation and Test performance (seed 42)")
ax.legend(frameon=False, ncol=2)
ax.spines[["top", "right"]].set_visible(False)

for i, value in enumerate(validation):
    ax.text(i - width / 2, value + 0.015, f"{value:.3f}",
            ha="center", va="bottom", fontsize=7, rotation=90)
for i, value in enumerate(test):
    ax.text(i + width / 2, value + 0.015, f"{value:.3f}",
            ha="center", va="bottom", fontsize=7, rotation=90)

fig.tight_layout()
save_figure(fig, "fig2_validation_vs_test")


# ------------------------------------------------------------
# Figure 3: Score quantiles for benign and malicious messages
# These are measured quantiles from the supplied run.
# They are NOT reconstructed raw score distributions.
# ------------------------------------------------------------
percentiles = np.array([1, 10, 25, 50, 75, 90, 99])

val_benign_q = np.array([
    0.00707739, 0.01119797, 0.01631997,
    0.02692985, 0.04546509, 0.07333849, 0.16695645
])
val_malicious_q = np.array([
    0.01010854, 0.02701717, 0.06658037,
    0.23654647, 0.59067762, 0.80386260, 0.94803994
])

test_benign_q = np.array([
    0.00648787, 0.01104659, 0.01558673,
    0.02627356, 0.04484368, 0.07102427, 0.15700691
])
test_malicious_q = np.array([
    0.00731394, 0.01680913, 0.03678671,
    0.12229836, 0.47931124, 0.79505305, 0.95060266
])

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharey=True)

axes[0].plot(percentiles, val_benign_q, marker="o", label="Benign")
axes[0].plot(percentiles, val_malicious_q, marker="s", label="Malicious")
axes[0].axhline(0.133019, linestyle="--",
                label="Validation-selected threshold")
axes[0].set_title("Validation")
axes[0].set_xlabel("Percentile")
axes[0].set_ylabel("Predicted malicious-class score")
axes[0].legend(frameon=False, fontsize=7)

axes[1].plot(percentiles, test_benign_q, marker="o", label="Benign")
axes[1].plot(percentiles, test_malicious_q, marker="s", label="Malicious")
axes[1].axhline(0.133019, linestyle="--",
                label="Threshold fixed from Validation")
axes[1].set_title("Test")
axes[1].set_xlabel("Percentile")
axes[1].legend(frameon=False, fontsize=7)

for ax in axes:
    ax.set_xticks(percentiles)
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Measured score quantiles by class (seed 42)", y=1.02)
fig.tight_layout()
save_figure(fig, "fig3_score_quantiles")


# ------------------------------------------------------------
# Figure 4: Test confusion matrix
# Rows = actual class; columns = predicted class.
# Class 0 = Benign; class 1 = Malicious.
# ------------------------------------------------------------
cm = np.array([
    [34939, 610],
    [4476, 4234],
])

fig, ax = plt.subplots(figsize=(4.8, 3.8))
im = ax.imshow(cm, interpolation="nearest", aspect="auto")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
             label="Number of messages")

ax.set(
    xticks=[0, 1],
    yticks=[0, 1],
    xticklabels=["Benign", "Malicious"],
    yticklabels=["Benign", "Malicious"],
    xlabel="Predicted class",
    ylabel="Actual class",
    title="Test confusion matrix (threshold = 0.133019)",
)

threshold_for_text = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, f"{cm[i, j]:,}",
            ha="center", va="center",
            color="white" if cm[i, j] > threshold_for_text else "black",
            fontsize=10,
        )

fig.tight_layout()
save_figure(fig, "fig4_test_confusion_matrix")

print(f"Figures saved to: {OUT.resolve()}")
for stem in [
    "fig1_intas_class_distribution",
    "fig2_validation_vs_test",
    "fig3_score_quantiles",
    "fig4_test_confusion_matrix",
]:
    print(f"  {stem}.pdf")
    print(f"  {stem}.png")
