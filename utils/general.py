import pandas as pd
import pickle
from config import (
    TRAIN_EVALUATE_MIN_RATINGS,
    PROJECT_DIRECTORY,
)
import utils.trust as trust_utils


def load_dataset():
    """
    Loads dataset and returns it as a pandas dataframe.
    """
    try:
        dataset_df = pickle.load(open(f"{PROJECT_DIRECTORY}/cache/dataset_df.pkl", "rb"))
    except FileNotFoundError:
        dataset_df = pd.read_csv(f"{PROJECT_DIRECTORY}/data/dataset.csv", parse_dates=[
            "date",
            "published_at"
        ])
        dataset_df["frequency"] = dataset_df["frequency"].apply(lambda x: [int(i) for i in x.split(";")])

        # Remove empty interactions
        dataset_df = dataset_df[
            (dataset_df["liked"] > 0)
            |
            (dataset_df["bookmarked"] > 0)
            |
            (dataset_df["consumed"] > 0)
            ].copy()

        pickle.dump(dataset_df, open(f"{PROJECT_DIRECTORY}/cache/dataset_df.pkl", "wb"))

    return dataset_df


def get_train_evaluation_split(
        evaluate_from,
        evaluate_to,
        dataset_df
):
    """
    Splits dataset into train and evaluation set.
    """
    train_df = __get_train_split(dataset_df, evaluate_from)
    evaluate_df = __get_evaluate_split(dataset_df, train_df, evaluate_from, evaluate_to)

    return train_df, evaluate_df


def __get_train_split(
        dataset_df,
        evaluate_from,
):
    """
    Gets train split
    """
    # Get train dataset
    train_df = dataset_df[dataset_df["date"] < evaluate_from].copy()
    train_df = trust_utils.attach_trust_score(train_df)

    train_unique_interactions_df = train_df.drop_duplicates(subset=["user_id", "subject_id"])
    train_user_counts = train_unique_interactions_df["user_id"].value_counts()
    train_valid_users = train_user_counts[train_user_counts >= TRAIN_EVALUATE_MIN_RATINGS].index

    selected_train_df = train_df[train_df["user_id"].isin(train_valid_users)]

    return selected_train_df


def __get_evaluate_split(
        dataset_df,
        train_df,
        evaluate_from,
        evaluate_to,
):
    """
    Gets evaluation split
    """
    # First, determine trustworthy users on which we are training with sufficient interactions count
    # so we don't consider them as cold start users.
    trustworthy_unique_interactions_df = train_df[
        train_df["trust_score"] >= .5].drop_duplicates(subset=["user_id", "subject_id"])
    trustworthy_user_counts = trustworthy_unique_interactions_df["user_id"].value_counts()
    evaluate_valid_users = trustworthy_user_counts[trustworthy_user_counts >= TRAIN_EVALUATE_MIN_RATINGS].index

    evaluation_df = dataset_df[
        (dataset_df["date"] >= evaluate_from)
        &
        (dataset_df["date"] <= evaluate_to)
        &
        (dataset_df["trust_score"] >= .5)
        &
        (dataset_df["published_at"] <= evaluate_from)
        ]

    evaluation_df = evaluation_df[evaluation_df["user_id"].isin(evaluate_valid_users)]

    # We can not predict cold start items either way...
    evaluation_df = evaluation_df[
        evaluation_df["subject_id"].isin(train_df["subject_id"])
    ]

    return evaluation_df
