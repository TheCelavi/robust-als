#!/usr/bin/env python
# coding: utf-8
import sys
import os

sys.path.extend([os.path.dirname(os.path.abspath(__file__))])

from config import (
    TRAIN_TARGETS,
    NUMBER_OF_PARAMETER_TUNINGS_PER_MODEL,
    TRAIN_EVALUATE_SPLITS,
    EXPERIMENT_NAME,
)
import gc
import utils.general as general_utils
import utils.trust as trust_utils
import utils.study as study_utils

##########################
#      Prepare data      #
##########################

dataset_df = general_utils.load_dataset()
dataset_df = trust_utils.attach_trust_score(dataset_df)

##########################
#   Train for each day   #
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

    for target in TRAIN_TARGETS:
        for i in range(NUMBER_OF_PARAMETER_TUNINGS_PER_MODEL):
            study = study_utils.search_parameters(
                df=train_df,
                use_bm25=target["use_bm25"],
                robust=target["robust"],
                num_users=num_users,
                num_items=num_items,
            )
            best_params = study.best_params

            study_utils.save_parameters(
                model=target["name"],
                split=split,
                study=study,
                experiment=EXPERIMENT_NAME
            )

        gc.collect()

    gc.collect()
