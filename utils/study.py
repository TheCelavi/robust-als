import pandas as pd
import gc
import numpy as np
import optuna
import csv
import os
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import ndcg_at_k
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import csr_matrix
from utils.score import get_score
from config import (
    TRAIN_EVALUATE_K,
    OPTUNA_N_TRIALS,
    OPTUNA_ALPHA_RANGE,
    OPTUNA_ITERATIONS_RANGE,
    OPTUNA_REGULARIZATION_RANGE,
    OPTUNA_BM25_K1_RANGE,
    OPTUNA_BM25_B_RANGE,
    TRAIN_CV_FOLDS,
    TRAIN_EXPANDING_WINDOW_SIZE,
    OPTUNA_TAU_RANGE,
    OPTUNA_N_STARTUP_TRIALS,
    OPTUNA_BETA_RANGE,
    OPTUNA_FACTORS_RANGE,
    PROJECT_DIRECTORY,
    OPTUNA_NUMBER_OF_JOBS,
)


def search_parameters(
        df,
        use_bm25=False,
        robust=False,
        num_users=None,
        num_items=None,
):
    """
    Tune parameters using TPE sampler.
    """
    def objective(trial):
        als_params = {
            "alpha": trial.suggest_categorical("alpha", OPTUNA_ALPHA_RANGE),
            "factors": trial.suggest_int("factors", *OPTUNA_FACTORS_RANGE),
            "iterations": trial.suggest_categorical("iterations", OPTUNA_ITERATIONS_RANGE),
            "regularization": trial.suggest_float("regularization", *OPTUNA_REGULARIZATION_RANGE, log=True),
        }
        beta = trial.suggest_float("beta", *OPTUNA_BETA_RANGE)
        tau = trial.suggest_float("tau", *OPTUNA_TAU_RANGE) if robust else None
        scores = []
        max_date = df["date"].max()

        for i in range(TRAIN_CV_FOLDS):
            test_to_date = max_date - pd.Timedelta(days=i * TRAIN_EXPANDING_WINDOW_SIZE)
            test_from_date = test_to_date - pd.Timedelta(days=TRAIN_EXPANDING_WINDOW_SIZE - 1)
            train_to_date = test_from_date - pd.Timedelta(days=1)

            train_df = df[df["date"] <= train_to_date]
            train_scores_df = get_score(
                df=train_df,
                beta=beta,
                decay=True,
                tau=tau,
            )

            validate_df = df[
                (df["date"] >= test_from_date)
                &
                (df["date"] <= test_to_date)
                ]
            validate_df = validate_df[
                validate_df["user_id"].isin(train_scores_df["user_id"])
                &
                validate_df["subject_id"].isin(train_scores_df["subject_id"])
                ]

            validate_scores_df = get_score(
                df=validate_df,
                beta=beta,
                decay=False,
                tau=True if robust else None,
            )

            train_matrix = csr_matrix(
                (train_scores_df["score"], (train_scores_df["user_id"], train_scores_df["subject_id"])),
                shape=(num_users, num_items),
            )
            validate_matrix = csr_matrix(
                (validate_scores_df["score"], (validate_scores_df["user_id"], validate_scores_df["subject_id"])),
                shape=(num_users, num_items),
            )

            if use_bm25:
                bm25_k1 = trial.suggest_categorical("bm25_k1", OPTUNA_BM25_K1_RANGE)
                bm25_b = trial.suggest_float("bm25_b", *OPTUNA_BM25_B_RANGE, log=True)
                train_matrix = bm25_weight(train_matrix, bm25_k1, bm25_b).tocsr()

            model = AlternatingLeastSquares(
                **als_params,
                use_gpu=True,
            )
            model.fit(train_matrix, show_progress=False)

            scores.append(ndcg_at_k(model, train_matrix, validate_matrix, TRAIN_EVALUATE_K))

            gc.collect()

        trial.set_user_attr("max_score", max(scores))
        trial.set_user_attr("stddev", np.std(scores))

        gc.collect()

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=OPTUNA_N_STARTUP_TRIALS)
    )

    study.optimize(
        objective,
        n_trials=OPTUNA_N_TRIALS,
        show_progress_bar=True,
        n_jobs=OPTUNA_NUMBER_OF_JOBS,
        callbacks=[LiveMonitoring(TRAIN_EVALUATE_K)]
    )

    return study


def save_parameters(
        model,
        split,
        study,
        experiment,
):
    """
    Save tuned parameters.
    """
    row = {
        "model": model,
        "split": split,
        f"train_ndcg@{TRAIN_EVALUATE_K}": round(study.best_value, 4),
        f"max_ndcg@{TRAIN_EVALUATE_K}": round(study.best_trial.user_attrs["max_score"], 4),
        f"stddev_ndcg@{TRAIN_EVALUATE_K}": round(study.best_trial.user_attrs["stddev"], 6),
        "alpha": study.best_params["alpha"],
        "iterations": study.best_params["iterations"],
        "regularization": study.best_params["regularization"],
        "factors": study.best_params["factors"],
        "bm25_k1": study.best_params.get("bm25_k1"),
        "bm25_b": study.best_params.get("bm25_b"),
        "beta": study.best_params.get("beta"),
        "tau": study.best_params.get("tau"),
    }

    file = f"{PROJECT_DIRECTORY}/parameters/{experiment}.csv"

    file_exists = os.path.isfile(file)

    with open(file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # Write header only if file is new
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


class LiveMonitoring:
    """
    Helper class to monitor progress.
    """
    def __init__(self, k, metric="NDCG"):
        self.previous_best = 0
        self.k = k
        self.metric = metric

    def __call__(self, study, trial):
        if self.previous_best < study.best_value:
            self.previous_best = study.best_value

            print("=" * 30)
            print(f"Trial: {study.best_trial.number}")
            print(f"{self.metric}@{self.k}: {study.best_value:.4f}")

            print("Parameters info:")

            for key, value in study.best_params.items():
                print(f"{key:20}: {value}")

            print("=" * 30)

            print(f"Max score: {study.best_trial.user_attrs['max_score']:.4f}")
            print(f"Score stddev: {study.best_trial.user_attrs['stddev']:.6f}")

            print("=" * 30)
