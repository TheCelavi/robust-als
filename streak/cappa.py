#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)
from statsmodels.stats.proportion import proportion_confint

sns.set_theme(style="white", context="paper", font_scale=1.2)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.titlesize": 13,
})

LABELS = ["Expected", "Outlier", "Unknown"]

# Load Excel
df = pd.read_excel("validation.xlsx")
columns = ["validator_1", "validator_2", "heuristic"]

print("Total streaks:", len(df))

# Inter-rater agreement

validator_1 = df["validator_1"]
validator_2 = df["validator_2"]

agreement = (validator_1 == validator_2).sum()
total = len(df)

raw_agreement = agreement / total

kappa = cohen_kappa_score(
    validator_1,
    validator_2,
    labels=LABELS,
)

ci_low, ci_high = proportion_confint(
    agreement,
    total,
    alpha=0.05,
    method="wilson",
)

print("\nINTER-RATER AGREEMENT")
print(f"Raw agreement: {raw_agreement:.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"Cohen's kappa: {kappa:.4f}")

matrix = confusion_matrix(validator_1, validator_2, labels=LABELS)

print("\nValidators confusion matrix:")
print(pd.DataFrame(
    matrix,
    index=LABELS,
    columns=LABELS,
))

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=1,
    linecolor="white",
    xticklabels=LABELS,
    yticklabels=LABELS,
    cbar_kws={"label": "Number of cases"},
    ax=ax
)
ax.set_title(
    "Human Inter-Rater Confusion Matrix\n"
    "Raw agreement = 82.67%, Cohen's κ = 0.729"
)
ax.set_xlabel("Validator 2")
ax.set_ylabel("Validator 1")

plt.tight_layout()
fig.savefig("interrater_confusion.png", bbox_inches="tight")
plt.show()

# Validators agreement by category

print("\nAGREEMENT BY HEURISTIC CATEGORY")

agreement_results = []

for label in LABELS:
    subset = df[df["heuristic"] == label]

    n = len(subset)
    matching = (
        subset["validator_1"] == subset["validator_2"]
    ).sum()

    low, high = proportion_confint(
        matching, n, alpha=0.05,
        method="wilson",
    )

    agreement = matching / n if n > 0 else 0

    print(
        f"{label}: {matching}/{n}, "
        f"agreement={matching/n:.3f}, "
        f"95% CI=[{low:.3f}, {high:.3f}]"
    )

    agreement_results.append({
        "label": label,
        "agreement": agreement,
        "low": low,
        "high": high,
        "n": n,
    })

labels = [r["label"] for r in agreement_results]
values = np.array([r["agreement"] for r in agreement_results])
lows = np.array([r["low"] for r in agreement_results])
highs = np.array([r["high"] for r in agreement_results])
counts = [r["n"] for r in agreement_results]

# Asymmetric error bars: distance from estimate to CI limits
errors = np.vstack([
    values - lows,
    highs - values,
])

colors = {
    "Expected": "#2E8B57",
    "Outlier": "#377EB8",
    "Unknown": "#E69F00",
}

bar_colors = [colors.get(label, "#777777") for label in labels]

fig, ax = plt.subplots(figsize=(8, 5.5))

x = np.arange(len(labels))

ax.bar(
    x,
    values,
    color=bar_colors,
    width=0.62,
    edgecolor="white",
    linewidth=1.2,
    zorder=2,
)

ax.errorbar(
    x,
    values,
    yerr=errors,
    fmt="none",
    ecolor="black",
    elinewidth=1.5,
    capsize=5,
    capthick=1.5,
    zorder=3,
)

# Annotate agreement, confidence interval, and sample size
for i, (value, low, high, n) in enumerate(
    zip(values, lows, highs, counts)
):
    ax.annotate(
        f"{value:.1%}\n95% CI [{low:.1%}, {high:.1%}]\nn={n}",
        xy=(i, high),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Inter-rater agreement")
ax.set_xlabel("Heuristic category")
ax.set_title("Inter-Rater Agreement by Heuristic Category")

ax.set_ylim(0, 1.22)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.2)])

ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    "agreement_by_heuristic_category.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# Consensus

consensus = df[
    df["validator_1"] == df["validator_2"]
].copy()

consensus["human_label"] = consensus["validator_1"]

print("\nCONSENSUS")
print(f"Agreed cases: {len(consensus)}/{total}")


consensus_counts = (
    consensus["human_label"]
    .value_counts()
    .reindex(LABELS, fill_value=0)
)

total_consensus = len(consensus)

colors = {
    "Expected": "#2E8B57",
    "Outlier": "#377EB8",
    "Unknown": "#E69F00",
}

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(
    consensus_counts.index,
    consensus_counts.values,
    color=[colors[label] for label in consensus_counts.index],
    width=0.62,
    edgecolor="white",
)

# Annotate counts and percentages
for bar, count in zip(bars, consensus_counts.values):
    percentage = count / total_consensus if total_consensus else 0

    ax.annotate(
        f"{count} ({percentage:.1%})",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
    )

ax.set_title(
    f"Human Consensus Label Distribution\n"
    f"Agreed cases: {total_consensus}/{total}"
)
ax.set_xlabel("Human consensus label")
ax.set_ylabel("Number of agreed cases")

ax.set_ylim(0, max(consensus_counts.max() * 1.18, 1))
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    "human_consensus_distribution.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# Heuristic vs human consensus

print("\nHEURISTIC VS HUMAN CONSENSUS")

matrix = pd.crosstab(
    consensus["heuristic"],
    consensus["human_label"],
).reindex(
    index=LABELS,
    columns=LABELS,
    fill_value=0,
)

print(matrix)

# Class-specific agreement, conditioning on validators having reached the same judgment.
for label in LABELS:
    subset = consensus[
        consensus["heuristic"] == label
    ]

    if len(subset) == 0:
        continue

    matches = (
        subset["human_label"] == label
    ).sum()

    low, high = proportion_confint(
        matches,
        len(subset),
        alpha=0.05,
        method="wilson",
    )

    print(
        f"{label}: {matches}/{len(subset)} "
        f"({matches/len(subset):.3f}), "
        f"95% CI=[{low:.3f}, {high:.3f}]"
    )


# Use the matrix already computed above
# Rows - heuristic labels
# Columns - human consensus labels

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Raw confusion matrix
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=1,
    linecolor="white",
    xticklabels=LABELS,
    yticklabels=LABELS,
    cbar_kws={"label": "Number of cases"},
    ax=axes[0],
)

axes[0].set_title("Heuristic vs Human Consensus\nRaw Counts")
axes[0].set_xlabel("Human consensus label")
axes[0].set_ylabel("Heuristic prediction")

# Row-normalized confusion matrix
row_totals = matrix.sum(axis=1).replace(0, np.nan)

normalized_matrix = matrix.div(row_totals, axis=0).fillna(0)

sns.heatmap(
    normalized_matrix,
    annot=True,
    fmt=".1%",
    cmap="Purples",
    vmin=0,
    vmax=1,
    linewidths=1,
    linecolor="white",
    xticklabels=LABELS,
    yticklabels=LABELS,
    cbar_kws={"label": "Proportion within heuristic category"},
    ax=axes[1],
)

axes[1].set_title("Heuristic vs Human Consensus\nRow-Normalized")
axes[1].set_xlabel("Human consensus label")
axes[1].set_ylabel("Heuristic prediction")

fig.suptitle(
    f"Heuristic Validation on Consensus Cases (n={len(consensus)})",
    fontsize=14,
    fontweight="bold",
)

plt.tight_layout()
plt.savefig(
    "heuristic_vs_human_consensus.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# Binary comparison, only expected/outlier heuristic cases with expected/outlier human consensus.
binary = consensus[
    consensus["heuristic"].isin(
        ["Expected", "Outlier"]
    )
    & consensus["human_label"].isin(
        ["Expected", "Outlier"]
    )
]

print("\nBINARY HEURISTIC COMPARISON")
print(f"Evaluated cases: {len(binary)}")


print(classification_report(
        binary["human_label"],
        binary["heuristic"],
        labels=["Outlier", "Expected"],
        zero_division=0,
))


labels = ["Outlier", "Expected"]

cm = confusion_matrix(
        binary["human_label"],
        binary["heuristic"],
        labels=labels
)

plt.figure(figsize=(7, 5))

sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
)

plt.title("Binary Heuristic vs Human Consensus")
plt.xlabel("Heuristic Prediction")
plt.ylabel("Human Consensus")
plt.tight_layout()
plt.show()

# 6. Disagreements

disagreements = df[
    df["validator_1"] != df["validator_2"]
].copy()

disagreements.to_excel(
    "validators_disagreements.xlsx",
    index=False,
)

print("\nSaved validators_disagreements.xlsx")
