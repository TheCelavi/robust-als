#!/usr/bin/env python
# coding: utf-8
import sys
import os

sys.path.extend([os.path.dirname(os.path.abspath(__file__))])

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pickle
from config import (
    LOGREG_HEURISTIC_RULES,
    TRAIN_EVALUATE_SPLITS, LOGREG_FEATURES, PROJECT_DIRECTORY,
)
from utils.general import load_dataset
from utils.streak import build_streaks

##########################
#      Prepare data      #
##########################
try:
    df = pickle.load(open(f"{PROJECT_DIRECTORY}/cache/streaks_df.pkl", "rb"))
except FileNotFoundError:
    dataset_df = load_dataset()
    dataset_df = dataset_df[dataset_df['date'] < TRAIN_EVALUATE_SPLITS["week_1"]["evaluate_from"]]
    # Load streaks
    df = build_streaks(dataset_df)
    pickle.dump(df, open(f"{PROJECT_DIRECTORY}/cache/streaks_df.pkl", "wb"))

##########################
#   Pseudo labeling      #
##########################
df["trust"] = None
df["trust"] = np.select(
    condlist=[
        (
            (df["max_consumed_hours_day_streak"] >= LOGREG_HEURISTIC_RULES["outliers"]["max_consumed_hours_day_streak"])
            |
            (
                    (df["average_consumed_hours_day_streak"] >= LOGREG_HEURISTIC_RULES["outliers"]["average_consumed_hours_day_streak"])
                    &
                    (df["streak_duration"] >= LOGREG_HEURISTIC_RULES["min_duration_for_average"])
            )
            |
            (df["bookmarked_per_day_streak"] >= LOGREG_HEURISTIC_RULES["outliers"]["bookmarked_per_day_streak"])
            |
            (df["liked_per_day_streak"] >= LOGREG_HEURISTIC_RULES["outliers"]["liked_per_day_streak"])
        ),
        (
            (df["max_consumed_hours_day_streak"] <= LOGREG_HEURISTIC_RULES["expected"]["max_consumed_hours_day_streak"])
            &
            (
                (
                    (df["average_consumed_hours_day_streak"] <= LOGREG_HEURISTIC_RULES["expected"]["average_consumed_hours_day_streak"])
                    &
                    (df["streak_duration"] >= LOGREG_HEURISTIC_RULES["min_duration_for_average"])
                )
                |
                (
                    (df["average_consumed_hours_day_streak"] <= LOGREG_HEURISTIC_RULES["expected"]["max_consumed_hours_day_streak"])
                    &
                    (df["streak_duration"] == 1)
                )
            )
            &
            (df["bookmarked_per_day_streak"] <= LOGREG_HEURISTIC_RULES["expected"]["bookmarked_per_day_streak"])
            &
            (df["liked_per_day_streak"] <= LOGREG_HEURISTIC_RULES["expected"]["bookmarked_per_day_streak"])
        )
    ],
    choicelist=[
        0,
        1
    ],
    default=None
)

# Remove unlabeled
df = df.dropna(subset=["trust"])
df["trust"] = df["trust"].astype(int)

##########################
#          Train         #
##########################

features = LOGREG_FEATURES

x = df[features]
y = df["trust"]  # fraudulent = 0, genuine = 1

# train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# resample unbalanced samples
smote = SMOTE(sampling_strategy="auto", random_state=42)
x_resampled, y_resampled = smote.fit_resample(x_train, y_train)

# train model
model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, solver="lbfgs"))
model.fit(x_resampled, y_resampled)

# calculate predictions
predictions = model.predict(x_test)
probabilities = model.predict_proba(x_test)[:, 1]

##########################
#         Report         #
##########################

roc_auc = roc_auc_score(y_test, probabilities)
precision, recall, _ = precision_recall_curve(y_test, probabilities)
pr_auc = auc(recall, precision)
accuracy = accuracy_score(y_test, predictions)

print(f"ROC AUC: {roc_auc:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Confusion matrix

conf_matrix = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Classification report

print(classification_report(y_test, predictions))

##########################
#        Save model      #
##########################

pickle.dump(model, open(f"{PROJECT_DIRECTORY}/model/logreg.pkl", "wb"))