import os
import csv
from config import PROJECT_DIRECTORY


def save_params_csv(
        split,
        study,
        filename = "default"
):
    path = f"{PROJECT_DIRECTORY}/rce/parameters/{filename}.csv"

    # Flatten the dictionary
    row = {
        "model": 'rce',
        "split": split,
        "train_ndcg@10": round(study.best_value, 4),
        "max_ndcg@10": round(study.best_trial.user_attrs["max_score"], 4),
        "stddev_ndcg@10": round(study.best_trial.user_attrs["stddev"], 6),
        "beta": study.best_params["beta"],
        "factors": study.best_params["factors"],
        "epochs": study.best_params.get("epochs"),
        "lr": study.best_params.get("lr"),
        "batch_size": study.best_params.get("batch_size")
    }

    file_exists = os.path.isfile(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # Write header only if file is new
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)