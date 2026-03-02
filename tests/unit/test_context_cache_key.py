"""Tests for cache key computation — determinism and collision resistance."""

from __future__ import annotations

from scout_ai.context.cache.key_strategy import compute_cache_key, compute_index_hash


class TestComputeCacheKey:
    """Cache keys should be deterministic and collision-resistant."""

    def test_deterministic(self) -> None:
        key1 = compute_cache_key("q1", "idx_abc", "gpt-4o")
        key2 = compute_cache_key("q1", "idx_abc", "gpt-4o")
        assert key1 == key2

    def test_different_inputs_different_keys(self) -> None:
        key1 = compute_cache_key("q1", "idx_abc", "gpt-4o")
        key2 = compute_cache_key("q2", "idx_abc", "gpt-4o")
        assert key1 != key2

    def test_model_affects_key(self) -> None:
        key1 = compute_cache_key("q1", "idx_abc", "gpt-4o")
        key2 = compute_cache_key("q1", "idx_abc", "claude-3-5-sonnet")
        assert key1 != key2

    def test_context_hash_affects_key(self) -> None:
        key1 = compute_cache_key("q1", "idx_abc", "gpt-4o", "ctx1")
        key2 = compute_cache_key("q1", "idx_abc", "gpt-4o", "ctx2")
        assert key1 != key2

    def test_returns_hex_string(self) -> None:
        key = compute_cache_key("q1", "idx_abc", "gpt-4o")
        assert len(key) == 64  # SHA-256 hex digest
        int(key, 16)  # Should not raise if valid hex


class TestComputeIndexHash:
    """Index hash should be stable and change only when doc changes."""

    def test_deterministic(self) -> None:
        class FakeIndex:
            doc_id = "doc1"
            total_pages = 10
            trees = [1, 2, 3]

        h1 = compute_index_hash(FakeIndex())  # type: ignore[arg-type]
        h2 = compute_index_hash(FakeIndex())  # type: ignore[arg-type]
        assert h1 == h2

    def test_different_docs(self) -> None:
        class Index1:
            doc_id = "doc1"
            total_pages = 10
            trees = [1, 2, 3]

        class Index2:
            doc_id = "doc2"
            total_pages = 10
            trees = [1, 2, 3]

        assert compute_index_hash(Index1()) != compute_index_hash(Index2())  # type: ignore[arg-type]

    def test_truncated_hash(self) -> None:
        class FakeIndex:
            doc_id = "doc1"
            total_pages = 10
            trees = [1, 2, 3]

        h = compute_index_hash(FakeIndex())  # type: ignore[arg-type]
        assert len(h) == 16  # Truncated to 16 chars
