import numpy as np
from config import LOGREG_HEURISTIC_RULES


def get_score(
        df,
        beta,
        decay=False,
        tau=None,
):
    """
    Builds score dataframe
    """
    df = df.copy()

    w_watch, w_bookmark, w_like = __compute_weights(beta - 2.0)

    df["watched"] = df["consumed"] * df["progress_ratio"]
    df["rewatched"] = df["consumed"] * (df["consumed_ratio"] - df["progress_ratio"])

    if tau is not None:
        if tau is True:
            df = df[df["trust_score"] >= .5]
        else:
            if decay:
                df["watched"] = np.where(
                    df["trust_score"] < .5,
                    df["watched"] * __spike_penalty(
                        df["consumed_hours_day"],
                        LOGREG_HEURISTIC_RULES["expected"]["max_consumed_hours_day_streak"],
                        LOGREG_HEURISTIC_RULES["outliers"]["max_consumed_hours_day_streak"],
                        tau=tau
                    ),
                    df["watched"]
                )
                df["rewatched"] = np.where(
                    df["trust_score"] < .5,
                    df["rewatched"] * __spike_penalty(
                        df["consumed_hours_day"],
                        LOGREG_HEURISTIC_RULES["expected"]["max_consumed_hours_day_streak"],
                        LOGREG_HEURISTIC_RULES["outliers"]["max_consumed_hours_day_streak"],
                        tau=tau
                    ),
                    df["rewatched"]
                )
                df["bookmarked"] = np.where(
                    df["trust_score"] < .5,
                    df["bookmarked"] * __spike_penalty(
                        df["bookmarked_day"],
                        LOGREG_HEURISTIC_RULES["expected"]["bookmarked_per_day_streak"],
                        LOGREG_HEURISTIC_RULES["outliers"]["bookmarked_per_day_streak"],
                        tau=tau
                    ),
                    df["bookmarked"]
                )

    if decay:
        df = __temporal_decay(df)

    df = df.groupby(["user_id", "subject_id"]).agg(
        user_id=("user_id", "first"),
        subject_id=("subject_id", "first"),
        watched=("watched", "sum"),
        rewatched=("rewatched", "sum"),
        liked=("liked", "sum"),
        bookmarked=("bookmarked", "sum"),
        duration=("duration", "first"),
    ).reset_index(drop=True)

    df["score"] = (
            w_watch * __duration_watched_score(df["watched"], df["duration"]).astype(float)
            +
            w_watch * __duration_watched_score(df["rewatched"], df["duration"]).astype(float)
            +
            w_bookmark * df["bookmarked"].astype(float)
            +
            w_like * df["liked"].astype(float)
    )

    # Remove zero scores, if exists.
    df = df[df["score"] > 0.0]

    return df


def __duration_watched_score(total_watch, duration):
    """
    Normalize watch time
    """
    return np.log(1 + total_watch / np.sqrt(duration))

def __temporal_decay(df, halftime = 90):
    """
    Apply temporal decay
    """
    # Calculate delta
    ref_date = df["date"].max()
    df["days_since"] = (ref_date - df["date"]).dt.days

    # Calculate decay
    lambda_decay = np.log(2) / halftime
    df["time_decay"] = np.exp(-lambda_decay * df["days_since"])

    # Apply time decay
    df["watched"] = df["watched"] * df["time_decay"]
    df["rewatched"] = df["rewatched"] * df["time_decay"]
    df["liked"] = df["liked"] * df["time_decay"]
    df["bookmarked"] = df["bookmarked"] * df["time_decay"]

    return df

def __spike_penalty(x, low, high, epsilon=.5, eta=.01, tau=0):
    """
    Downweight suspicious interactions, or nullify them.
    """
    center = (high + low) / 2
    steepness = (high - low) / (2 * np.log((1 - eta) / eta))
    score = 1 - 1 / (1 + np.exp(-(x - center) / steepness))

    return np.where(
        score > tau,
        epsilon + (1 - epsilon) * score,
        0
    )


def __compute_weights(beta):
    """
    Calculate weights using Bezier quadratic functions.
    """
    # Ensure beta is a NumPy array for vectorized operations
    beta = np.asarray(beta)

    # Determine the 'dominates' condition based on beta values
    dominates = np.where(beta < 0, 'w_bookmark', 'w_like')

    normalized = np.abs(beta)

    # Calculate w_watch for all beta values
    w_watch = (1 - normalized) ** 2

    # Initialize w_bookmark and w_like arrays
    w_bookmark = np.zeros_like(normalized)
    w_like = np.zeros_like(normalized)

    # When dominates == 'w_bookmark', calculate w_bookmark and w_like
    w_bookmark[dominates == 'w_bookmark'] = 2 * normalized[dominates == 'w_bookmark'] * (
            1 - normalized[dominates == 'w_bookmark'])
    w_like[dominates == 'w_bookmark'] = normalized[dominates == 'w_bookmark'] ** 2

    # When dominates == 'w_like', calculate w_like and w_bookmark
    w_like[dominates == 'w_like'] = 2 * normalized[dominates == 'w_like'] * (1 - normalized[dominates == 'w_like'])
    w_bookmark[dominates == 'w_like'] = normalized[dominates == 'w_like'] ** 2

    return w_watch, w_bookmark, w_like
