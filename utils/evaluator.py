import math


class Evaluator:
    """
    Helper class to evaluate metrics@k.
    """
    def __init__(self, catalogue_size, metrics=None, k=None):
        if k is None:
            k = [5, 10, 20]
        if metrics is None:
            metrics = ["ndcg", "precision", "recall", "f1", "hitrate", "coverage"]
        self.prediction = []
        self.actual = []
        self.metrics = metrics
        self.k = k
        self.catalogue_size = catalogue_size

    def add(self, prediction, actual):
        self.prediction.append(prediction)
        self.actual.append(actual)

    def evaluate(self):
        results = {}

        for k in self.k:
            agg = {m: 0.0 for m in self.metrics}
            catalogue = []

            results[f"k={k}"] = {}

            for prediction, actual in zip(self.prediction, self.actual):
                if "ndcg" in self.metrics:
                    agg["ndcg"] += self.__ndcg(actual, prediction, k)

                if "precision" in self.metrics:
                    agg["precision"] += self.__precision(actual, prediction, k)

                if "recall" in self.metrics:
                    agg["recall"] += self.__recall(actual, prediction, k)

                if "hitrate" in self.metrics:
                    agg["hitrate"] += self.__hitrate(actual, prediction, k)

                if "f1" in self.metrics:
                    agg["f1"] += self.__f1_score(actual, prediction, k)

                if "coverage" in self.metrics:
                    catalogue.append(prediction[:k])

            n_users = len(self.prediction)

            for m in self.metrics:
                if m == "coverage":
                    continue

                results[f"k={k}"][f"{m}@{k}"] = agg[m] / n_users if n_users > 0 else 0.0

            if "coverage" in self.metrics:
                total = self.__unique_items(catalogue)
                results[f"k={k}"][f"coverage@{k}"] = total / self.catalogue_size

        return results

    @staticmethod
    def __hitrate(prediction, actual, k):
        """
        Compute Hit Rate@k metric.
        """

        actual_set = set(actual)

        if 0 == len(actual_set):
            return 0

        top_k = prediction[:k]

        return int(any(item in actual_set for item in top_k))

    @staticmethod
    def __recall(actual: list[int], predicted: list[int], k: int) -> float:
        """
        Calculate the recall@k metric.
        """
        actual_set = set(actual)

        if 0 == len(actual_set):
            return 0

        top_k_predictions = set(predicted[:k])
        count_relevant_in_top_k = len(actual_set.intersection(top_k_predictions))
        return count_relevant_in_top_k / float(len(actual_set))

    @staticmethod
    def __precision(actual: list[int], predicted: list[int], k: int) -> float:
        """
        Calculate the precision@k metric.
        """
        actual_set = set(actual)

        if 0 == len(actual_set):
            return 0

        top_k_predictions = set(predicted[:k])
        count_relevant_in_top_k = len(actual_set.intersection(top_k_predictions))
        return count_relevant_in_top_k / float(k)

    @staticmethod
    def __f1_score(actual: list[int], predicted: list[int], k: int) -> float:
        """
        Calculate the F1-score @k metric.
        """
        recall_score = Evaluator.__recall(actual, predicted, k)
        precision_score = Evaluator.__precision(actual, predicted, k)

        if recall_score == 0 or precision_score == 0:
            return float(0)

        return 2 * (recall_score * precision_score) / (recall_score + precision_score)

    @staticmethod
    def __ndcg(actual: list[int], predicted: list[int], k: int) -> float:
        """
        Computes the Normalized Discounted Cumulative Gain (nDCG) at a specified rank `k`.
        """
        actual_set = set(actual)

        # discounted cumulative gain
        # `i+2` due to zero indexing
        dcg = sum([1.0 / math.log2(i + 2) for i in range(k) if predicted[i] in actual_set])
        # ideal discounted cumulative gain (ie. perfect results returned)
        idcg = sum([1.0 / math.log2(i + 2) for i in range(min(k, len(actual_set)))])

        if idcg == 0:
            return float(0)

        return dcg / idcg

    @staticmethod
    def __unique_items(predictions):
        return len(set(item for user_preds in predictions for item in user_preds))
