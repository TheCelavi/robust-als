import pickle
from config import (
    LOGREG_FEATURES,
    PROJECT_DIRECTORY,
)
from utils.streak import process_streaks


def attach_trust_score(df):
    """
    Attach trust_score column to dataframe.
    """
    KEEP = {
        "consumed_hours_day",
        "bookmarked_day",
        "trust_score",
    }

    # Remove previously attached trust score and streak values, if exist
    df = df.drop(columns=list(KEEP) + ["min_date", "max_date"] + LOGREG_FEATURES, errors="ignore")

    df = process_streaks(df)
    logreg = pickle.load(open(f"{PROJECT_DIRECTORY}/model/logreg.pkl", "rb"))

    df["trust_score"] = logreg.predict_proba(df[LOGREG_FEATURES])[:, 1]

    filtered = [f for f in LOGREG_FEATURES if f not in KEEP]

    df.drop(columns=filtered, inplace=True)

    return df