import math
from collections import Counter, defaultdict
from bisect import bisect_left, bisect_right
from scipy.stats import spearmanr
from utils.evaluator import Evaluator


class BiasAwareEvaluator(Evaluator):

    BASE_METRICS = {
        "ndcg", "precision", "recall",
        "f1", "hitrate", "coverage"
    }

    BIAS_METRICS = {
        "reachable_coverage",
        "adjusted_gini",
        "long_tail_coverage",
        "long_tail_share",
        "arp",
        "arp_percentile",
        "popularity_correlation",
    }

    def __init__(
        self,
        catalogue_size,
        item_popularity,
        metrics=None,
        k=None,
        long_tail_fraction=0.8,
    ):
        if metrics is None:
            metrics = [
                "ndcg", "precision", "recall", "coverage",
                *sorted(self.BIAS_METRICS),
            ]

        unknown = set(metrics) - (
            self.BASE_METRICS | self.BIAS_METRICS
        )
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}")

        super().__init__(
            catalogue_size,
            metrics=[m for m in metrics if m in self.BASE_METRICS],
            k=k,
        )

        self.bias_metrics = set(metrics) & self.BIAS_METRICS
        self.item_popularity = item_popularity
        self.long_tail_fraction = long_tail_fraction
        self.accessible = []

    def add(self, prediction, actual, accessible_items):
        accessible_items = set(accessible_items)

        if not set(prediction).issubset(accessible_items):
            raise ValueError(
                "Predictions contain inaccessible items"
            )

        if len(prediction) != len(set(prediction)):
            raise ValueError("Duplicate recommended items")

        super().add(prediction, actual)
        self.accessible.append(accessible_items)

    def evaluate(self):
        results = super().evaluate()

        reachable = set().union(*self.accessible) if self.accessible else set()
        long_tail = self.__long_tail_items(reachable)

        for k in self.k:
            result = results[f"k={k}"]

            exposure = self.__exposure(k)
            expected = self.__expected_exposure(k)

            if "reachable_coverage" in self.bias_metrics:
                result[f"reachable_coverage@{k}"] = (
                    self.__reachable_coverage(exposure, reachable)
                )

            if "adjusted_gini" in self.bias_metrics:
                result[f"adjusted_gini@{k}"] = (
                    self.__adjusted_gini(exposure, expected)
                )

            if "long_tail_coverage" in self.bias_metrics:
                result[f"long_tail_coverage@{k}"] = (
                    self.__long_tail_coverage(exposure, long_tail)
                )

            if "long_tail_share" in self.bias_metrics:
                result[f"long_tail_share@{k}"] = (
                    self.__long_tail_share(exposure, long_tail)
                )

            if "arp" in self.bias_metrics:
                result[f"arp@{k}"] = self.__arp(exposure)

            if "arp_percentile" in self.bias_metrics:
                result[f"arp_percentile@{k}"] = (
                    self.__arp_percentile(k)
                )

            if "popularity_correlation" in self.bias_metrics:
                result[f"popularity_correlation@{k}"] = (
                    self.__popularity_correlation(exposure, expected)
                )

        return results

    # --- Shared calculations ---

    def __exposure(self, k):
        return Counter(
            item
            for prediction in self.prediction
            for item in prediction[:k]
        )

    def __expected_exposure(self, k):
        expected = defaultdict(float)

        for prediction, accessible in zip(
            self.prediction, self.accessible
        ):
            if not accessible:
                continue

            # Uniform random recommendation baseline.
            probability = len(prediction[:k]) / len(accessible)

            for item in accessible:
                expected[item] += probability

        return expected

    def __long_tail_items(self, reachable):
        if not reachable:
            return set()

        popularity = sorted(
            self.item_popularity.get(item, 0)
            for item in reachable
        )

        index = math.ceil(
            len(popularity) * self.long_tail_fraction
        ) - 1

        threshold = popularity[index]

        return {
            item for item in reachable
            if self.item_popularity.get(item, 0) <= threshold
        }

    # --- Individual metrics ---

    @staticmethod
    def __reachable_coverage(exposure, reachable):
        return (
            len(set(exposure) & reachable) / len(reachable)
            if reachable else 0.0
        )

    @staticmethod
    def __adjusted_gini(exposure, expected):
        pairs = sorted(
            (
                exposure.get(item, 0) / opportunity,
                opportunity,
            )
            for item, opportunity in expected.items()
            if opportunity > 0
        )

        cumulative_weight = 0.0
        cumulative_value = 0.0
        numerator = 0.0

        for ratio, weight in pairs:
            numerator += weight * (
                ratio * cumulative_weight - cumulative_value
            )
            cumulative_weight += weight
            cumulative_value += ratio * weight

        denominator = cumulative_weight * cumulative_value

        return (
            numerator / denominator
            if denominator > 0 else float("nan")
        )

    @staticmethod
    def __long_tail_coverage(exposure, long_tail):
        return (
            len(set(exposure) & long_tail) / len(long_tail)
            if long_tail else 0.0
        )

    @staticmethod
    def __long_tail_share(exposure, long_tail):
        total = sum(exposure.values())

        return (
            sum(exposure[item] for item in long_tail) / total
            if total else 0.0
        )

    def __arp(self, exposure):
        total = sum(exposure.values())

        return (
            sum(
                self.item_popularity.get(item, 0) * count
                for item, count in exposure.items()
            ) / total
            if total else float("nan")
        )

    def __arp_percentile(self, k):
        percentiles = []

        for prediction, accessible in zip(
            self.prediction, self.accessible
        ):
            if not accessible:
                continue

            popularity = sorted(
                self.item_popularity.get(item, 0)
                for item in accessible
            )

            n = len(popularity)

            for item in prediction[:k]:
                value = self.item_popularity.get(item, 0)

                left = bisect_left(popularity, value)
                right = bisect_right(popularity, value)

                # Midrank treatment of tied popularity.
                percentile = (
                    (left + right - 1) / (2 * (n - 1))
                    if n > 1 else 0.5
                )

                percentiles.append(percentile)

        return (
            sum(percentiles) / len(percentiles)
            if percentiles else float("nan")
        )

    def __popularity_correlation(self, exposure, expected):
        items = [
            item for item, opportunity in expected.items()
            if opportunity > 0
        ]

        popularity = [
            self.item_popularity.get(item, 0)
            for item in items
        ]

        adjusted_exposure = [
            exposure.get(item, 0) / expected[item]
            for item in items
        ]

        if (
            len(items) < 2
            or len(set(popularity)) < 2
            or len(set(adjusted_exposure)) < 2
        ):
            return float("nan")

        return float(
            spearmanr(popularity, adjusted_exposure).statistic
        )
