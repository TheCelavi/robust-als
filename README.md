# Robust ALS

Contains experiment for robust ALS in incentive-driven environment.

Depends on:

- Conda environment
- Numpy
- Pandas
- Implicit (GPU implementation

## Scripts:

- `config.py` - General configuration for experiments
- `logreg.py` - Trains logreg for trust estimator
- `tune.py` - Executes hyperparameter tuning
- `test.py` - Trains model and evaluates against unseen interactions

## Dataset

Available in `data` directory. Unpack zip archive with two files, `data/acl.csv` and `data/dataset.csv`.