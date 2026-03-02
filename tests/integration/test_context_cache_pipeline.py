"""Integration test: full pipeline with result caching."""

from __future__ import annotations

from scout_ai.context.cache.key_strategy import compute_cache_key, compute_index_hash
from scout_ai.context.cache.memory import MemoryCache
from tests.fakes.fake_context_cache import FakeContextCache


class TestCachedExtractionPipeline:
    """Verify caching integrates correctly with extraction pipeline."""

    async def test_cache_stores_and_retrieves_results(self) -> None:
        """Cache should store extraction results and return them on hit."""
        cache = MemoryCache(max_entries=100)

        # Simulate storing extraction result
        key = compute_cache_key("q1", "idx_abc", "gpt-4o")
        result = {"answer": "Hypertension", "confidence": 0.95}
        await cache.put(key, result, ttl_seconds=3600)

        # Should retrieve same result
        cached = await cache.get(key)
        assert cached == result

    async def test_cache_miss_returns_none(self) -> None:
        """Cache miss should return None, triggering fresh extraction."""
        cache = MemoryCache()
        key = compute_cache_key("q_new", "idx_abc", "gpt-4o")
        assert await cache.get(key) is None

    async def test_different_models_different_cache_keys(self) -> None:
        """Different model names should produce different cache keys."""
        cache = MemoryCache()
        key1 = compute_cache_key("q1", "idx_abc", "gpt-4o")
        key2 = compute_cache_key("q1", "idx_abc", "claude-3-5-sonnet")

        await cache.put(key1, {"answer": "A"})
        await cache.put(key2, {"answer": "B"})

        assert (await cache.get(key1))["answer"] == "A"
        assert (await cache.get(key2))["answer"] == "B"

    async def test_fake_cache_integration(self) -> None:
        """FakeContextCache should work as drop-in replacement."""
        cache = FakeContextCache()
        key = compute_cache_key("q1", "idx_abc", "gpt-4o")
        await cache.put(key, {"result": True})
        assert await cache.get(key) == {"result": True}
        await cache.invalidate(key)
        assert await cache.get(key) is None

    def test_index_hash_stability(self) -> None:
        """Same index should always produce same hash."""

        class FakeIndex:
            doc_id = "doc1"
            total_pages = 50
            trees = list(range(5))

        h1 = compute_index_hash(FakeIndex())  # type: ignore[arg-type]
        h2 = compute_index_hash(FakeIndex())  # type: ignore[arg-type]
        assert h1 == h2

    async def test_lru_eviction_with_extraction_results(self) -> None:
        """LRU should evict oldest entries when cache is full."""
        cache = MemoryCache(max_entries=3)
        for i in range(5):
            key = compute_cache_key(f"q{i}", "idx", "model")
            await cache.put(key, {"i": i})

        # First two should be evicted
        key0 = compute_cache_key("q0", "idx", "model")
        key4 = compute_cache_key("q4", "idx", "model")
        assert await cache.get(key0) is None
        assert (await cache.get(key4))["i"] == 4
