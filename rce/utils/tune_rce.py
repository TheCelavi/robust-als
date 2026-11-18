import optuna
import numpy as np
import pandas as pd

from rce.utils.ndcg_at_10 import ndcg_at_10
from rce.utils.rce import fit_rce
from rce.utils.to_sparse_matrix import to_sparse_matrix
from utils.score import get_score
from utils.study import LiveMonitoring
from config import (
    TRAIN_CV_FOLDS,
    TRAIN_EXPANDING_WINDOW_SIZE,
    OPTUNA_NUMBER_OF_JOBS,
)


def tune_rce(
        df,
        num_users=None,
        num_items=None,
        n_trials=150,
        seed=42
):
    def objective(trial):
        params = {
            "beta": trial.suggest_float("beta", 0.05, 1.0),
            "factors": trial.suggest_categorical("factors", [1]),
            "epochs": trial.suggest_categorical("epochs", [5, 10, 20, 40]),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [1024, 2048, 4096]),
        }

        scores = []
        max_date = df["date"].max()

        for i in range(TRAIN_CV_FOLDS):
            test_to_date = max_date - pd.Timedelta(days=i * TRAIN_EXPANDING_WINDOW_SIZE)
            test_from_date = test_to_date - pd.Timedelta(days=TRAIN_EXPANDING_WINDOW_SIZE - 1)
            train_to_date = test_from_date - pd.Timedelta(days=1)

            train_df = df[df["date"] <= train_to_date]
            train_scores_df = get_score(
                df=train_df,
                beta=None,
                decay=False,
                tau=None,
            )

            validate_df = df[
                (df["date"] >= test_from_date)
                &
                (df["date"] <= test_to_date)
                ]
            validate_df = validate_df[
                validate_df["user_idx"].isin(train_scores_df["user_idx"])
                &
                validate_df["subject_idx"].isin(train_scores_df["subject_idx"])
                ]
            validate_scores_df = get_score(
                df=validate_df,
                beta=None,
                decay=False,
                tau=None,
            )

            X_train_fold = to_sparse_matrix(train_scores_df, n_users=num_users, n_items=num_items)
            X_valid_fold = to_sparse_matrix(validate_scores_df, n_users=num_users, n_items=num_items)

            model = fit_rce(
                X_train_fold,
                **params,
                seed=seed
            )

            score = ndcg_at_10(
                model,
                X_train_fold,
                X_valid_fold
            )

            scores.append(score)

        trial.set_user_attr("fold_ndcg", scores)

        trial.set_user_attr("max_score", max(scores))
        trial.set_user_attr("stddev", np.std(scores))

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=OPTUNA_NUMBER_OF_JOBS,
        callbacks=[LiveMonitoring(10)]
    )

    return study
