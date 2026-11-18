import csv
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from scipy.stats import t

folder = Path(__file__).parent
weeks = defaultdict(list)

with (folder / "source.csv").open(newline="") as f:
    for row in csv.DictReader(f):
        weeks[row["split"]].append({
            "ndcg@10": float(row["r2-ndcg@10"]) - float(row["v1-ndcg@10"]),
            "recall@20": float(row["r2-recall@20"]) - float(row["v1-recall@20"]),
        })

with (folder / "paired_by_seed.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["week", "seed", "delta_ndcg@10", "delta_recall@20"])
    for week, rows in weeks.items():
        for seed, row in enumerate(rows, 1):
            writer.writerow([week, seed, f'{row["ndcg@10"]:.6f}', f'{row["recall@20"]:.6f}'])

with (folder / "paired_by_week.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["week", "n", "mean_delta_ndcg@10", "ci95_lower_ndcg@10", "ci95_upper_ndcg@10", "mean_delta_recall@20", "ci95_lower_recall@20", "ci95_upper_recall@20"])
    for week, rows in weeks.items():
        values = [[row[metric] for row in rows] for metric in ("ndcg@10", "recall@20")]
        result = []
        for v in values:
            m = mean(v)
            margin = t.ppf(0.975, len(v) - 1) * stdev(v) / sqrt(len(v))
            result.extend([f"{m:.6f}", f"{m - margin:.6f}", f"{m + margin:.6f}"])
        writer.writerow([week, len(rows), *result])