#!/usr/bin/env python
# coding: utf-8
import sys
import os

sys.path.extend([
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
])

from rce.utils.rce import fit_rce, recommend
from rce.utils.to_sparse_matrix import to_sparse_matrix
from utils.score import get_score
import gc
import pickle
import pandas as pd
from config import (
    TRAIN_EVALUATE_SPLITS,
    PROJECT_DIRECTORY,
)
import utils.general as general_utils
import utils.acl as acl_utils
import utils.evaluate as evaluate_utils
import utils.trust as trust_utils
from utils.bias_aware_evaluator import BiasAwareEvaluator
import torch
import random

torch.set_num_threads(1)
##########################
#      Prepare data      #
##########################

EXPERIMENT = 'factor_1'

try:
    dataset_df = pickle.load(open(f"{PROJECT_DIRECTORY}/cache/dataset_df.pkl", "rb"))
except FileNotFoundError:
    # Load dataset
    dataset_df, user_uniques, subject_uniques = general_utils.load_dataset()
    # Attach trust score to interactions
    dataset_df = trust_utils.attach_trust_score(dataset_df)
    # Save for latter
    pickle.dump(dataset_df, open(f"{PROJECT_DIRECTORY}/cache/dataset_df.pkl", "wb"))

##########################
#    Load parameters     #
##########################

parameters_df = pd.read_csv(f"{PROJECT_DIRECTORY}/rce/parameters/{EXPERIMENT}.csv")

##########################
#  Train for each split  #
##########################

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
    train_scores_df = get_score(
        df=train_df,
        beta=None,
        decay=False,
    )

    # Extract required parameters for given split
    split_parameters_df = parameters_df[parameters_df["split"] == split]

    # Determine enabled subjects
    enabled_subjects = set(dataset_df.loc[dataset_df["enabled"] == 1, "subject_idx"].astype(int).tolist())
    # Determine catalogue size
    catalogue_size = train_df["subject_id"].nunique()

    # Popularity
    glob_item_popularity = general_utils.get_popularity_count(evaluate_from, dataset_df)

    for index, row in split_parameters_df.iterrows():
        for i in range(10):
            matrix = to_sparse_matrix(train_scores_df, n_users=num_users, n_items=num_items)
            model = fit_rce(
                matrix,
                beta=row["beta"],
                factors=row["factors"],
                epochs=row["epochs"],
                batch_size=row["batch_size"],
                lr=row["lr"],
                seed=random.randint(0, 1000000)
            )

            actual = evaluate_utils.build_actual_map(evaluate_df)
            # Get ACL list for given date
            acl = acl_utils.get_acl(
                date_from=evaluate_from,
                date_to=evaluate_to,
                evaluate_df=evaluate_df,
            )

            glob_pop_evaluator = BiasAwareEvaluator(
                catalogue_size=catalogue_size,
                item_popularity=glob_item_popularity,
            )

            for user_id, interacted in actual.items():
                unseen = evaluate_utils.get_unrated_items(matrix, user_id)
                expected = list(set(unseen) & set(interacted))
                accessible = list(set(acl[user_id]) & set(enabled_subjects))

                predictions = recommend(model, user_id, accessible, matrix[user_id].indices, 20)

                glob_pop_evaluator.add(predictions, expected, accessible)

            evaluate_utils.save_report(
                model=row["model"],
                split=split,
                metrics=glob_pop_evaluator.evaluate(),
                sparsity=0,
                ratings=0,
                experiment=EXPERIMENT,
            )

            gc.collect()
