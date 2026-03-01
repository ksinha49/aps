"""Metric computation functions for retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievalMetrics:
    """Aggregated retrieval evaluation metrics."""

    total_cases: int = 0
    recall_at_5: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    per_case: list[dict[str, object]] = field(default_factory=list)


def compute_recall_at_k(
    expected_node_ids: list[str],
    retrieved_node_ids: list[str],
    k: int = 5,
) -> float:
    """Proportion of expected nodes found in top-k retrieved.

    Args:
        expected_node_ids: Ground-truth node IDs that should be retrieved.
        retrieved_node_ids: Node IDs returned by the retrieval system, in rank order.
        k: Number of top results to consider.

    Returns:
        Recall score between 0.0 and 1.0. Returns 1.0 if expected_node_ids is empty.
    """
    if not expected_node_ids:
        return 1.0
    top_k = set(retrieved_node_ids[:k])
    hits = len(set(expected_node_ids) & top_k)
    return hits / len(expected_node_ids)


def compute_mrr(
    expected_node_ids: list[str],
    retrieved_node_ids: list[str],
) -> float:
    """Reciprocal rank of first relevant result.

    Args:
        expected_node_ids: Ground-truth node IDs that are considered relevant.
        retrieved_node_ids: Node IDs returned by the retrieval system, in rank order.

    Returns:
        Reciprocal rank (1/position) of the first relevant result, or 0.0 if none found.
    """
    expected_set = set(expected_node_ids)
    for i, node_id in enumerate(retrieved_node_ids, start=1):
        if node_id in expected_set:
            return 1.0 / i
    return 0.0


def evaluate_retrieval_cases(
    cases: list[dict[str, object]],
    k: int = 5,
) -> RetrievalMetrics:
    """Evaluate a batch of retrieval cases and compute aggregate metrics.

    Each case dict must contain:
        - "expected_node_ids": list[str]
        - "retrieved_node_ids": list[str]
        - "query": str (optional, for per-case reporting)

    Args:
        cases: List of case dicts with expected and retrieved node IDs.
        k: Number of top results to consider for recall.

    Returns:
        Aggregated RetrievalMetrics with per-case breakdowns.
    """
    if not cases:
        return RetrievalMetrics()

    total_recall = 0.0
    total_mrr = 0.0
    per_case: list[dict[str, object]] = []

    for case in cases:
        expected: list[str] = case.get("expected_node_ids", [])  # type: ignore[assignment]
        retrieved: list[str] = case.get("retrieved_node_ids", [])  # type: ignore[assignment]
        query: str = case.get("query", "")  # type: ignore[assignment]

        recall = compute_recall_at_k(expected, retrieved, k)
        mrr = compute_mrr(expected, retrieved)

        total_recall += recall
        total_mrr += mrr

        per_case.append({
            "query": query,
            "recall_at_k": recall,
            "mrr": mrr,
            "expected": expected,
            "retrieved_top_k": retrieved[:k],
        })

    n = len(cases)
    return RetrievalMetrics(
        total_cases=n,
        recall_at_5=total_recall / n,
        mrr=total_mrr / n,
        per_case=per_case,
    )
