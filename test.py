#!/usr/bin/env python
# coding: utf-8
import sys
import os

sys.path.extend([os.path.dirname(os.path.abspath(__file__))])

import gc
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import csr_matrix
from config import (
    TRAIN_EVALUATE_SPLITS,
    PROJECT_DIRECTORY,
    EXPERIMENT_NAME,
    NUMBER_OF_TRAINED_MODELS_PER_PARAMETERS,
)
import utils.general as general_utils
import utils.acl as acl_utils
import utils.evaluate as evaluate_utils
import utils.trust as trust_utils
import utils.score as score_utils
from utils.evaluator import Evaluator

##########################
#      Prepare data      #
##########################

dataset_df = general_utils.load_dataset()
dataset_df = trust_utils.attach_trust_score(dataset_df)

##########################
#    Load parameters     #
##########################

parameters_df = pd.read_csv(f"{PROJECT_DIRECTORY}/parameters/{EXPERIMENT_NAME}.csv")

##########################
#  Train for each split  #
##########################

num_users = dataset_df["user_id"].max() + 1
num_items = dataset_df["subject_id"].max() + 1

for split, config in TRAIN_EVALUATE_SPLITS.items():
    print(f"Processing split: {split}")

    evaluate_from = config["evaluate_from"]
    evaluate_to = config["evaluate_to"]

    train_df, evaluate_df = general_utils.get_train_evaluation_split(
        evaluate_from=evaluate_from,
        evaluate_to=evaluate_to,
        dataset_df=dataset_df,
    )

    # Extract required parameters for given split
    split_parameters_df = parameters_df[parameters_df["split"] == split]

    # Get actual interactions for given dates
    actual = evaluate_utils.build_actual_map(evaluate_df)
    # Get ACL list for given date
    acl = acl_utils.get_acl(
        date_from=evaluate_from,
        date_to=evaluate_to,
        evaluate_df=evaluate_df,
    )
    # Determine enabled subjects
    enabled_subjects = set(dataset_df.loc[dataset_df["enabled"] == 1, "subject_id"].astype(int).tolist())
    # Determine catalogue size
    catalogue_size = train_df["subject_id"].nunique()

    for index, row in split_parameters_df.iterrows():
        for i in range(NUMBER_OF_TRAINED_MODELS_PER_PARAMETERS):
            model = AlternatingLeastSquares(
                alpha=row["alpha"],
                factors=row["factors"],
                iterations=row["iterations"],
                regularization=row["regularization"],
                use_gpu=False,
            )

            robust = row["model"] in ["robust", "robust_bm25"]
            use_bm25 = row["model"] in ["vanilla_bm25", "robust_bm25"]

            train_scores_df = score_utils.get_score(
                df=train_df,
                beta=row["beta"],
                decay=True,
                tau=row["tau"] if robust else None,
            )

            matrix = csr_matrix(
                (train_scores_df["score"], (train_scores_df["user_id"], train_scores_df["subject_id"])),
                shape=(num_users, num_items),
            )

            if use_bm25:
                matrix = bm25_weight(
                    matrix,
                    row["bm25_k1"],
                    row["bm25_b"]
                ).tocsr()

            evaluator = Evaluator(catalogue_size=catalogue_size)
            sparsity, ratings = evaluate_utils.get_matrix_sparsity(matrix)

            model.fit(matrix)

            for user_id, interacted in actual.items():
                unseen = evaluate_utils.get_unrated_items(matrix, user_id)
                expected = list(set(unseen) & set(interacted))
                accessible = list(set(acl[user_id]) & set(enabled_subjects))

                predictions, _ = model.recommend(
                    user_id,
                    matrix[user_id],
                    N=20,
                    items=accessible,
                    filter_already_liked_items=True
                )

                evaluator.add(predictions, expected)

            evaluate_utils.save_report(
                model=row["model"],
                split=split,
                metrics=evaluator.evaluate(),
                sparsity=sparsity,
                ratings=ratings,
                experiment=EXPERIMENT_NAME
            )

            gc.collect()
