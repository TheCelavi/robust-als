#!/usr/bin/env python
# coding: utf-8
import os

sys.path.extend([
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
])

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from config import (
    PROJECT_DIRECTORY,
)

DIRECTORY = f"{PROJECT_DIRECTORY}/wilcoxon"

vanilla = pd.read_csv(f"{DIRECTORY}/vanilla_1_factor_user.csv")
robust = pd.read_csv(f"{DIRECTORY}/robust_2_factor_user.csv")
metrics = ['ndcg@10', 'recall@20']
keys = ['split', 'run', 'user']
merged = vanilla.merge(robust, on=keys, suffixes=('_vanilla', '_robust'), validate='one_to_one')
results = []

for week, frame in merged.groupby('split', sort=True):
    for metric in metrics:
        user_values = frame.groupby('user')[[f'{metric}_robust', f'{metric}_vanilla']].mean()
        difference = np.round((user_values[f'{metric}_robust'] - user_values[f'{metric}_vanilla']).to_numpy(), 8)
        test = wilcoxon(difference, zero_method='wilcox', alternative='two-sided', method='auto')
        results.append({
            'week': week,
            'metric': metric,
            'users': len(difference),
            'improved': int(sum(difference > 0)),
            'worsened': int(sum(difference < 0)),
            'unchanged': int(sum(difference == 0)),
            'mean_difference': difference.mean(),
            'median_difference': float(np.median(difference)),
            'statistic': test.statistic, 'p_value': test.pvalue
        })

summary = pd.DataFrame(results)
summary['holm_p_value'] = multipletests(summary.p_value, method='holm')[1]

summary.to_csv(f"{DIRECTORY}wilcoxon_user_results.csv", index=False)

print(summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
