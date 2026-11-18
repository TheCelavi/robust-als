import numpy as np
import pandas as pd
import threadpoolctl
import optuna
import os
import warnings
from optuna.exceptions import ExperimentalWarning

pd.set_option('display.max_columns', None)
threadpoolctl.threadpool_limits(1, "blas")
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings(
    "ignore",
    category=ExperimentalWarning,
    module="optuna"
)

# Number of thread to use, get from system number of CPUs
THREAD_COUNT = os.cpu_count() or 8

LOGREG_FEATURES = [
    "streak_duration",
    "consumed_hours_streak",
    "liked_streak",
    "liked_per_day_streak",
    "bookmarked_streak",
    "bookmarked_per_day_streak",
    "interacted_streak",
    "max_consumed_hours_day_streak",
    "average_consumed_hours_day_streak",
    "std_dev_consumed_hours_day_streak",
    "average_completion_streak",
    "std_dev_completion_streak",
    "unique_watched_streak",
]

LOGREG_HEURISTIC_RULES = {
    "min_duration_for_average": 2,
    "outliers": {
        "max_consumed_hours_day_streak": 18.0,
        "average_consumed_hours_day_streak": 12.0,
        "bookmarked_per_day_streak": 100,
        "liked_per_day_streak": 100,
    },
    "expected": {
        "max_consumed_hours_day_streak": 6.0,
        "average_consumed_hours_day_streak": 4.0,
        "bookmarked_per_day_streak": 30,
        "liked_per_day_streak": 30,
    }
}

# Train split
TRAIN_EVALUATE_K = 10
TRAIN_CV_FOLDS = 3
TRAIN_EXPANDING_WINDOW_SIZE = 7
TRAIN_EVALUATE_MIN_RATINGS = 10

# How many times to tune and train same model to get median metrics
NUMBER_OF_PARAMETER_TUNINGS_PER_MODEL = 20
NUMBER_OF_TRAINED_MODELS_PER_PARAMETERS = 10

# Hyperparameter search
OPTUNA_N_TRIALS = 150
OPTUNA_N_STARTUP_TRIALS = 15
OPTUNA_ALPHA_RANGE = [1, 5, 10, 20, 40, 80, 100]
OPTUNA_FACTORS_RANGE = (2, 16)
OPTUNA_ITERATIONS_RANGE = [5, 10, 20, 30, 50, 80, 100]
OPTUNA_REGULARIZATION_RANGE = (0.001, 0.1)
OPTUNA_BM25_K1_RANGE = [20, 50, 70, 100, 120, 150, 170, 200]
OPTUNA_BM25_B_RANGE = (0.1, 1)
OPTUNA_TAU_RANGE = (0.00001, 0.1)
OPTUNA_BETA_RANGE = (1.0, 3.0) # This translates to [-1,1], Optuna can not work with negative values.

# High number of jobs make process unstable, number depends on your HW.
OPTUNA_NUMBER_OF_JOBS = 5

# Evaluation
EVALUATION_PROGRESS_THRESHOLDS = {
    30: 1,
    60: 0.9,
    300: 0.8,
    900: 0.5,
    1800: 0.4,
    3600: 0.15,
    np.inf: 0.1
}

TRAIN_TARGETS = [
    {
        "name": "vanilla",
        "robust": False,
        "use_bm25": False,
    },
    {
        "name": "vanilla_bm25",
        "robust": False,
        "use_bm25": True,
    },
    {
        "name": "robust",
        "robust": True,
        "use_bm25": False,
    },
    {
        "name": "robust_bm25",
        "robust": True,
        "use_bm25": True,
    },
]

TRAIN_EVALUATE_SPLITS = {
    "week_1": {
        "evaluate_from": pd.to_datetime("2025-06-23"),
        "evaluate_to": pd.to_datetime("2025-06-29"),
    },
    "week_2": {
        "evaluate_from": pd.to_datetime("2025-06-30"),
        "evaluate_to": pd.to_datetime("2025-07-06"),
    },
    "week_3": {
        "evaluate_from": pd.to_datetime("2025-07-07"),
        "evaluate_to": pd.to_datetime("2025-07-13"),
    },
    "week_4": {
        "evaluate_from": pd.to_datetime("2025-07-13"),
        "evaluate_to": pd.to_datetime("2025-07-20"),
    },
}

PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Set experiment name.
EXPERIMENT_NAME = 'ablation_no_penalty'
