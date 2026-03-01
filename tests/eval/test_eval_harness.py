"""Tests for the retrieval evaluation harness metrics."""

from __future__ import annotations

from tests.eval.retrieval_evaluator import compute_mrr, compute_recall_at_k


class TestRecallAtK:
    """Tests for compute_recall_at_k."""

    def test_recall_at_5_perfect(self) -> None:
        """All expected nodes found in top 5 results gives recall 1.0."""
        expected = ["n1", "n2", "n3"]
        retrieved = ["n1", "n2", "n3", "n4", "n5"]
        assert compute_recall_at_k(expected, retrieved, k=5) == 1.0

    def test_recall_at_5_partial(self) -> None:
        """Only 1 of 2 expected nodes in top 5 gives recall 0.5."""
        expected = ["n1", "n2"]
        retrieved = ["n1", "n99", "n98", "n97", "n96"]
        assert compute_recall_at_k(expected, retrieved, k=5) == 0.5

    def test_recall_at_5_none(self) -> None:
        """No expected nodes in top 5 gives recall 0.0."""
        expected = ["n1", "n2"]
        retrieved = ["n10", "n20", "n30", "n40", "n50"]
        assert compute_recall_at_k(expected, retrieved, k=5) == 0.0


class TestMRR:
    """Tests for compute_mrr (Mean Reciprocal Rank)."""

    def test_mrr_first(self) -> None:
        """First result is relevant gives MRR 1.0."""
        expected = ["n1"]
        retrieved = ["n1", "n2", "n3"]
        assert compute_mrr(expected, retrieved) == 1.0

    def test_mrr_second(self) -> None:
        """Second result is relevant gives MRR 0.5."""
        expected = ["n2"]
        retrieved = ["n1", "n2", "n3"]
        assert compute_mrr(expected, retrieved) == 0.5

    def test_mrr_not_found(self) -> None:
        """No relevant results gives MRR 0.0."""
        expected = ["n99"]
        retrieved = ["n1", "n2", "n3"]
        assert compute_mrr(expected, retrieved) == 0.0
