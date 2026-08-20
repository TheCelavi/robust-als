import pandas as pd
import numpy as np
import pickle
from config import PROJECT_DIRECTORY


def get_acl(
        date_from,
        date_to,
        evaluate_df,
):
    """
    Get ACL list for given date range.

    Use cache for not recalculating ACL list for same date range.
    """
    suffix = date_from.strftime("%Y-%m-%d") + "-" + date_to.strftime("%Y-%m-%d")
    path = f"{PROJECT_DIRECTORY}/cache/acl-{suffix}.pkl"

    try:
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        acl_map = __load_acl_map(date_from, date_to, evaluate_df)
        pickle.dump(acl_map, open(path, "wb"))
        return acl_map


def __load_acl_map(date_from, date_to, candidates_df):
    """
    Load ACL list for given date range.
    """
    acl_df = pd.read_csv(f"{PROJECT_DIRECTORY}/data/acl.csv", parse_dates=["created_at", "valid_to"])

    # Apply trick for unlimited access to simplify filtering logic
    acl_df["valid_to"] = acl_df["valid_to"].fillna(pd.to_datetime("2040-01-01"))

    # Keep only ACL applicable within evaluation period
    acl_df = acl_df[
        (acl_df["created_at"] <= date_to)
        &
        (acl_df["valid_to"] >= date_from)
        ]

    # Decrease ACL dataset to potential evaluation users, use less memory.
    acl_df = acl_df[acl_df["user_id"].isin(candidates_df["user_id"])]

    acl_map = (
        acl_df.groupby("user_id")["subject_id"]
        .apply(list)
        .to_dict()
    )

    for user_id in acl_map:
        acl_map[user_id] = np.array([acl_map[user_id]]).astype(int).tolist()[0]

    return acl_map
