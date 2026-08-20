import numpy as np
import csv
import os
from config import (
    EVALUATION_PROGRESS_THRESHOLDS,
    PROJECT_DIRECTORY,
)


def build_actual_map(evaluation_df):
    """
    Build actual map from evaluation dataframe
    """
    actual_df = evaluation_df.groupby(["user_id", "subject_id"]).agg(
        user_id=("user_id", "first"),
        subject_id=("subject_id", "first"),
        progress=(
            "frequency",
            lambda x: np.any(
                np.stack([np.fromiter(i, dtype=int) for i in x]) > 0,
                axis=0
            ).sum()
        ),
        consumed=("consumed", "sum"),
        liked=("liked", "sum"),
        bookmarked=("bookmarked", "sum"),
        duration=("duration", "first"),
    ).reset_index(drop=True)

    # Calculate progress ratio, unique watch progress
    actual_df["progress_ratio"] = actual_df["progress"] / actual_df["duration"]

    # Calculate progress cut off
    bounds = np.array(sorted(EVALUATION_PROGRESS_THRESHOLDS.keys()))
    thresholds = np.array([EVALUATION_PROGRESS_THRESHOLDS[b] for b in bounds])
    threshold_index = np.searchsorted(bounds, actual_df["duration"].values, side="right")

    actual_df["threshold"] = thresholds[threshold_index]

    actual_relevant_df = actual_df[actual_df["progress_ratio"] >= actual_df["threshold"]]

    actual_map = (
        actual_relevant_df.groupby("user_id")["subject_id"]
        .apply(list)
        .to_dict()
    )

    return actual_map


def get_unrated_items(user_item_matrix, user_id):
    """
    Returns list of unrated item ids for a given user.
    """

    rated_items = user_item_matrix[user_id].indices
    all_items = np.arange(user_item_matrix.shape[1])

    unrated_items = np.setdiff1d(all_items, rated_items)

    return unrated_items.tolist()


def save_report(
        model,
        split,
        metrics,
        sparsity,
        ratings,
        experiment,
):
    """
    Save evaluation report
    """
    row = {
        "model": model,
        "split": split,
        "sparsity": sparsity,
        "ratings": ratings,
    }

    for k_group, values in metrics.items():
        for metric_name, metric_value in values.items():
            row[metric_name] = round(metric_value, 4)

    file = f"{PROJECT_DIRECTORY}/report/{experiment}.csv"

    file_exists = os.path.isfile(file)

    with open(file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # Write header only if file is new
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def get_matrix_sparsity(matrix):
    """
    Returns sparsity of matrix
    """
    mat = matrix.copy()
    non_empty_rows = np.diff(mat.indptr) > 0
    mat = mat[non_empty_rows]
    non_empty_cols = np.array(mat.sum(axis=0)).ravel() != 0
    mat = mat[:, non_empty_cols]
    total_elements = mat.shape[0] * mat.shape[1]
    non_zero = mat.nnz

    sparsity = 1.0 - (non_zero / total_elements)

    return sparsity, round((1 - sparsity) * total_elements)
