import pandas as pd
import numpy as np
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from rich.progress import Progress
from config import (
    THREAD_COUNT,
    LOGREG_FEATURES,
)


def process_streaks(train_df):
    """
    Process streaks and attach required data for logreg to the dataset.
    """
    streaks_df = build_streaks(train_df)

    daily_consumption_df = train_df.groupby(["user_id", "date"]).agg(
        user_id=("user_id", "first"),
        date=("date", "first"),
        consumed_hours_day=("consumed_hours", "sum"),
        bookmarked_day=("bookmarked", "sum"),
        liked_day=("liked", "sum"),
    ).reset_index(drop=True)

    train_df = train_df.merge(
        daily_consumption_df,
        how="left",
        on=["user_id", "date"]
    ).reset_index(drop=True)

    train_df = train_df.merge(
        streaks_df[["user_id", "min_date", "max_date"] + LOGREG_FEATURES],
        on="user_id",
        how="left"
    ).reset_index(drop=True)

    train_df["in_range"] = (
            (train_df["date"] >= train_df["min_date"]) &
            (train_df["date"] <= train_df["max_date"])
    )

    train_df = train_df[train_df["in_range"] == True]
    train_df = train_df.drop(columns=["in_range"])

    return train_df


def build_streaks(df):
    """
    Build streaks for logreg.
    """
    subtasks = __create_subtask_df(df, "user_id", 100)
    results = []

    with Progress() as progress:
        task = progress.add_task("Building streaks...", total=len(subtasks))

        with ProcessPoolExecutor(max_workers=THREAD_COUNT) as executor:
            futures = [executor.submit(__process_batch, batch) for batch in subtasks]

            for future in as_completed(futures):
                try:
                    result_df = future.result()
                    results.append(result_df)
                    progress.update(task, advance=1)
                except Exception as e:
                    print("Worker failed:", repr(e))
                    raise

    result = pd.concat(results, ignore_index=True)

    return result


def __process_batch(batch_df):
    """
    Process batch of streaks.
    """
    users = batch_df["user_id"].unique().tolist()
    results = []

    for user_id in users:
        user_interactions_df = batch_df[batch_df["user_id"] == user_id]
        streak_df = __build_streaks_for_user(user_interactions_df)
        results.append(streak_df)

    return pd.concat(results, ignore_index=True)


def __build_streaks_for_user(records):
    """
    Build streaks for single user.
    """
    records = records.sort_values(by="date", ascending=True)
    records["diff"] = records["date"].diff().dt.days
    records["group"] = (records["diff"] > 1).cumsum()
    dfs = [g.drop(columns=["diff", "group"]) for _, g in records.groupby("group")]
    results = []

    for streak_source_df in dfs:
        daily_df = streak_source_df.groupby("date").agg(
            user_id=("user_id", "first"),
            date=("date", "first"),
            consumed_hours=("consumed_hours", "sum"),
        ).reset_index(drop=True)

        daily_df = daily_df.sort_values(by=["user_id", "date"])

        daily_df = daily_df.groupby("user_id").agg(
            user_id=("user_id", "first"),
            max_consumed_hours_day_streak=("consumed_hours", "max"),
            average_consumed_hours_day_streak=("consumed_hours", "mean"),
            std_dev_consumed_hours_day_streak=("consumed_hours", "std"),
            # daily_consumption_in_streak=("consumed_hours", list)
        ).reset_index(drop=True)

        progress_df = streak_source_df.groupby(["user_id", "subject_id"]).agg(
            user_id=("user_id", "first"),
            subject_id=("subject_id", "first"),
            duration=("duration", "first"),
            completion=("frequency", __completion_watched),
        ).reset_index(drop=True)

        progress_df = progress_df[progress_df["completion"] > 0]
        progress_df["completion"] = progress_df["completion"] / progress_df["duration"]

        progress_df = progress_df.groupby(["user_id"]).agg(
            user_id=("user_id", "first"),
            average_completion_streak=("completion", "mean"),
            std_dev_completion_streak=("completion", "std"),
            unique_watched_streak=("subject_id", "nunique"),
        ).reset_index(drop=True)

        streak_df = streak_source_df.groupby("user_id").agg(
            user_id=("user_id", "first"),
            min_date=("date", "min"),
            max_date=("date", "max"),
            streak_duration=("date", "nunique"),
            consumed_hours_streak=("consumed_hours", "sum"),
            liked_streak=("liked", "sum"),
            bookmarked_streak=("bookmarked", "sum"),
            interacted_streak=("subject_id", "nunique"),
        ).reset_index(drop=True)

        combined_df = streak_df.merge(daily_df, how="left", on=["user_id"])
        combined_df = combined_df.merge(progress_df, how="left", on=["user_id"])

        combined_df["std_dev_consumed_hours_day_streak"] = combined_df["std_dev_consumed_hours_day_streak"].fillna(0)
        combined_df["std_dev_completion_streak"] = combined_df["std_dev_completion_streak"].fillna(0)
        combined_df["unique_watched_streak"] = combined_df["unique_watched_streak"].fillna(0).astype(int)
        combined_df["average_completion_streak"] = combined_df["average_completion_streak"].fillna(0)
        combined_df["liked_streak"] = combined_df["liked_streak"].fillna(0).astype(int)
        combined_df["liked_per_day_streak"] = combined_df["liked_streak"] / combined_df["streak_duration"]
        combined_df["bookmarked_streak"] = combined_df["bookmarked_streak"].fillna(0).astype(int)
        combined_df["bookmarked_per_day_streak"] = combined_df["bookmarked_streak"] / combined_df["streak_duration"]

        results.append(combined_df)

    return pd.concat(results, ignore_index=True)


def __create_subtask_df(df, group_by, nb_chunks=THREAD_COUNT):
    """
    Create subtasks dataframe for parallelization.
    """
    sub_df_sizes = df.groupby(group_by).size()
    sub_df_sizes = sub_df_sizes.sort_values(ascending=False)
    chunks = [[] for _ in range(nb_chunks)]
    chunk_sizes = [0] * nb_chunks

    # Greedy balancing
    for sub_df, size in sub_df_sizes.items():
        idx = chunk_sizes.index(min(chunk_sizes))
        chunks[idx].append(sub_df)
        chunk_sizes[idx] += size

    df_chunks = [df[df[group_by].isin(chunks)] for chunks in chunks]

    return df_chunks


def __completion_watched(series):
    """
    Calculate completion from frequency.
    """
    arr = np.asarray(series.tolist())
    return np.count_nonzero(arr.any(axis=0))
