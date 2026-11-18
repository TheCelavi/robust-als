#!/usr/bin/env python
# coding: utf-8
import sys
import os

sys.path.extend([
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
])

import pickle
import utils.general as general_utils
import utils.trust as trust_utils
from rce.utils.tune_rce import tune_rce
import torch
from rce.utils.write_csv import save_params_csv
from config import (
    PROJECT_DIRECTORY,
    TRAIN_EVALUATE_SPLITS,
    NUMBER_OF_PARAMETER_TUNINGS_PER_MODEL,
)

torch.set_num_threads(1)

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("PyTorch CUDA:", torch.version.cuda)
print("CPU threads:", torch.get_num_threads())

try:
    dataset_df = pickle.load(open(f"{PROJECT_DIRECTORY}/cache/dataset_df.pkl", "rb"))
except FileNotFoundError:
    # Load dataset
    dataset_df, user_uniques, subject_uniques = general_utils.load_dataset()
    # Attach trust score to interactions
    dataset_df = trust_utils.attach_trust_score(dataset_df)
    # Save for latter
    pickle.dump(dataset_df, open(f"{PROJECT_DIRECTORY}/cache/dataset_df.pkl", "wb"))

num_users = dataset_df["user_id"].nunique() + 1
num_items = dataset_df["subject_id"].nunique() + 1

for split, config in TRAIN_EVALUATE_SPLITS.items():
    print(f"Processing split: {split}")

    evaluate_from = config["evaluate_from"]
    evaluate_to = config["evaluate_to"]

    train_df, evaluate_df = general_utils.get_train_evaluation_split(
        evaluate_from=evaluate_from,
        evaluate_to=evaluate_to,
        dataset_df=dataset_df,
    )

    for i in range(NUMBER_OF_PARAMETER_TUNINGS_PER_MODEL):
        study = tune_rce(train_df, num_users, num_items)
        best_params = study.best_params

        save_params_csv(
            split=split,
            study=study,
            filename=split
        )

        print(best_params)
