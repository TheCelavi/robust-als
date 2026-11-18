#!/usr/bin/env python
# coding: utf-8
"""
Assess probabilities on unknown streaks.
"""
import os
import pickle
import sys

sys.path.extend([
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
])

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    LOGREG_HEURISTIC_RULES,
    TRAIN_EVALUATE_SPLITS,
    LOGREG_FEATURES,
    PROJECT_DIRECTORY,
)
from utils.general import load_dataset
from utils.streak import build_streaks

SEEDS = range(42, 52)
OUTPUT_DIR = f"{PROJECT_DIRECTORY}/streak"

# Prepare data; preserve existing streak-cache workflow.
cache_path = f"{PROJECT_DIRECTORY}/cache/streaks_df.pkl"
try:
    with open(cache_path, "rb") as f:
        df = pickle.load(f)
except FileNotFoundError:
    dataset_df = load_dataset()
    dataset_df = dataset_df[
        dataset_df["date"] < TRAIN_EVALUATE_SPLITS["week_1"]["evaluate_from"]
        ]
    df = build_streaks(dataset_df)
    df = df.sort_values(
        by=["min_date", "max_date", "user_id"]
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(df, f)

# Pseudo labeling: 0 = outlier, 1 = expected, NaN = unknown.
rules = LOGREG_HEURISTIC_RULES
outlier = (
        (df["max_consumed_hours_day_streak"] >= rules["outliers"]["max_consumed_hours_day_streak"])
        | (
                (df["average_consumed_hours_day_streak"] >= rules["outliers"]["average_consumed_hours_day_streak"])
                & (df["streak_duration"] >= rules["min_duration_for_average"])
        )
        | (df["bookmarked_per_day_streak"] >= rules["outliers"]["bookmarked_per_day_streak"])
        | (df["liked_per_day_streak"] >= rules["outliers"]["liked_per_day_streak"])
)
expected = (
        (df["max_consumed_hours_day_streak"] <= rules["expected"]["max_consumed_hours_day_streak"])
        & (
                (
                        (df["average_consumed_hours_day_streak"] <= rules["expected"][
                            "average_consumed_hours_day_streak"])
                        & (df["streak_duration"] >= rules["min_duration_for_average"])
                )
                | (
                        (df["average_consumed_hours_day_streak"] <= rules["expected"]["max_consumed_hours_day_streak"])
                        & (df["streak_duration"] == 1)
                )
        )
        & (df["bookmarked_per_day_streak"] <= rules["expected"]["bookmarked_per_day_streak"])
        & (df["liked_per_day_streak"] <= rules["expected"]["bookmarked_per_day_streak"])
)
df = df.copy()
df["trust"] = np.select([outlier, expected], [0, 1], default=np.nan)

# Keep the unknown streaks before dropping unlabeled rows.
unknown_df = df.loc[df["trust"].isna()].copy()
labeled_df = df.loc[df["trust"].notna()].copy()
labeled_df["trust"] = labeled_df["trust"].astype(int)
print(f"Labeled streaks: {len(labeled_df):,}; Unknown streaks: {len(unknown_df):,}")

features = LOGREG_FEATURES
x = labeled_df[features]
y = labeled_df["trust"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Keep the original fitting procedure exactly as supplied.
smote = SMOTE(sampling_strategy="auto", random_state=42)
x_resampled, y_resampled = smote.fit_resample(x_train, y_train)
model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, solver="lbfgs"))
model.fit(x_resampled, y_resampled)

predictions = model.predict(x_test)
probabilities = model.predict_proba(x_test)[:, 1]  # P(expected)
precision, recall, _ = precision_recall_curve(y_test, probabilities)
print(f"ROC AUC: {roc_auc_score(y_test, probabilities):.6f}")
print(f"PR AUC: {auc(recall, precision):.6f}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.6f}")
print(f"Brier score: {brier_score_loss(y_test, probabilities):.8f}")
print(f"Log loss: {log_loss(y_test, probabilities):.8f}")
print(classification_report(y_test, predictions))

ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, predictions)).plot(cmap="Blues")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=200)
plt.close()

dsp_exp = CalibrationDisplay.from_predictions(y_test, probabilities, n_bins=5, strategy="uniform")
dsp_exp.line_.set_linestyle("None")
plt.title("Calibration: Expected probabilities")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/calibration_expected.png", dpi=200)
plt.close()

dsp_out = CalibrationDisplay.from_predictions(1 - y_test, 1 - probabilities, n_bins=5, strategy="uniform")
dsp_out.line_.set_linestyle("None")
plt.title("Calibration: Outlier probabilities")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/calibration_outlier.png", dpi=200)
plt.close()

# Score the same unknown streaks with the original fitted model.
x_unknown = unknown_df[features]
trust_scores = model.predict_proba(x_unknown)[:, 1]
print("\nUnknown-streak trust-score percentiles (original model):")
for p, v in zip([0, 5, 25, 50, 75, 95, 100], np.percentile(trust_scores, [0, 5, 25, 50, 75, 95, 100])):
    print(f"  {p:>3}th: {v:.6f}")

plt.figure(figsize=(8, 5))
plt.hist(trust_scores, bins=np.linspace(0, 1, 31), edgecolor="black")
plt.xlabel("Predicted trust score P(Expected)")
plt.ylabel("Number of Unknown streaks")
plt.title("Trust scores for Unknown streaks")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/unknown_trust_histogram.png", dpi=200)
plt.close()

# Stability assessment: fixed train/test split, varying SMOTE/model seeds.
# Stability is NOT a calibration measurement.
score_runs = []
for seed in SEEDS:
    xs, ys = SMOTE(sampling_strategy="auto", random_state=seed).fit_resample(x_train, y_train)
    candidate = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, solver="lbfgs"))
    candidate.fit(xs, ys)
    score_runs.append(candidate.predict_proba(x_unknown)[:, 1])
score_runs = np.asarray(score_runs)  # shape: (10, n_unknown)
score_mean = score_runs.mean(axis=0)
score_sd = score_runs.std(axis=0, ddof=1)
score_range = score_runs.max(axis=0) - score_runs.min(axis=0)
print("\nAcross-seed stability of Unknown trust scores:")
print(f"  Mean per-streak SD: {score_sd.mean():.6f}")
print(f"  Median per-streak SD: {np.median(score_sd):.6f}")
print(f"  95th percentile SD: {np.percentile(score_sd, 95):.6f}")
print(f"  Mean per-streak range: {score_range.mean():.6f}")

plt.figure(figsize=(8, 5))
plt.hist(score_sd, bins=30, edgecolor="black")
plt.xlabel("SD of predicted trust scores across 10 random seeds")
plt.ylabel("Number of Unknown streaks")
plt.title("Prediction stability on Unknown streaks")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/unknown_trust_stability.png", dpi=200)
plt.close()

# Keep original row index for matching the results back to streaks.
result = unknown_df.copy()
result["predicted_trust"] = trust_scores
result["trust_mean_10_seeds"] = score_mean
result["trust_sd_10_seeds"] = score_sd
result["trust_range_10_seeds"] = score_range
result.to_csv(f"{OUTPUT_DIR}/unknown_trust_scores.csv", index=True)

summary = pd.DataFrame({
    "metric": [
        "n_unknown", "mean_trust", "median_trust", "pct_trust_below_0_1",
        "pct_trust_between_0_1_and_0_9", "pct_trust_above_0_9",
        "mean_seed_sd", "median_seed_sd", "p95_seed_sd",
    ],
    "value": [
        len(trust_scores), trust_scores.mean(), np.median(trust_scores),
        100 * np.mean(trust_scores < 0.1),
        100 * np.mean((trust_scores >= 0.1) & (trust_scores <= 0.9)),
        100 * np.mean(trust_scores > 0.9),
        score_sd.mean(), np.median(score_sd), np.percentile(score_sd, 95),
    ],
})
summary.to_csv(f"{OUTPUT_DIR}/unknown_trust_summary.csv", index=False)
print("\nSummary:\n", summary.to_string(index=False))
print(f"\nPlots and CSV files saved to: {OUTPUT_DIR}")
