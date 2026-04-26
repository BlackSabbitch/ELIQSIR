import math

class MetricsEngine:
    """
    Mathematical implementations of Information Retrieval (IR) metrics.
    """
    @staticmethod
    def precision(relevant: set, retrieved: list) -> float:
        if not retrieved: return 0.0
        return len(relevant.intersection(retrieved)) / len(retrieved)

    @staticmethod
    def recall(relevant: set, retrieved: list) -> float:
        if not relevant: return 0.0
        return len(relevant.intersection(retrieved)) / len(relevant)

    @staticmethod
    def f1_score(relevant: set, retrieved: list) -> float:
        p = MetricsEngine.precision(relevant, retrieved)
        r = MetricsEngine.recall(relevant, retrieved)
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    @staticmethod
    def precision_at_k(relevant: set, retrieved: list, k: int) -> float:
        return MetricsEngine.precision(relevant, retrieved[:k])

    @staticmethod
    def ndcg_at_k(relevant: set, retrieved: list, k: int) -> float:
        retrieved_k = retrieved[:k]
        dcg = sum([1.0 / math.log2(i + 2) for i, doc in enumerate(retrieved_k) if doc in relevant])
        idcg = sum([1.0 / math.log2(i + 2) for i in range(min(len(relevant), k))])
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def average_precision(relevant: set, retrieved: list) -> float:
        hits = 0
        sum_precisions = 0.0
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                hits += 1
                sum_precisions += hits / (i + 1)
        return sum_precisions / len(relevant) if relevant else 0.0