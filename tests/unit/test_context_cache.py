"""Tests for context cache backends — memory LRU, TTL, put/get/invalidate."""

from __future__ import annotations

import time

from scout_ai.context.cache.memory import MemoryCache
from scout_ai.context.models import CacheEntry


class TestCacheEntry:
    """CacheEntry should track TTL expiry correctly."""

    def test_not_expired_when_no_ttl(self) -> None:
        entry = CacheEntry(key="k", value="v", ttl_seconds=0)
        assert entry.is_expired is False

    def test_not_expired_within_ttl(self) -> None:
        entry = CacheEntry(key="k", value="v", ttl_seconds=3600)
        assert entry.is_expired is False

    def test_expired_after_ttl(self) -> None:
        entry = CacheEntry(
            key="k",
            value="v",
            created_at=time.time() - 10,
            ttl_seconds=5,
        )
        assert entry.is_expired is True


class TestMemoryCache:
    """MemoryCache should implement LRU eviction and TTL expiry."""

    async def test_put_and_get(self) -> None:
        cache = MemoryCache()
        await cache.put("key1", {"answer": "42"})
        result = await cache.get("key1")
        assert result == {"answer": "42"}

    async def test_get_miss_returns_none(self) -> None:
        cache = MemoryCache()
        assert await cache.get("nonexistent") is None

    async def test_invalidate(self) -> None:
        cache = MemoryCache()
        await cache.put("key1", "value1")
        await cache.invalidate("key1")
        assert await cache.get("key1") is None

    async def test_clear(self) -> None:
        cache = MemoryCache()
        await cache.put("a", 1)
        await cache.put("b", 2)
        await cache.clear()
        assert await cache.get("a") is None
        assert await cache.get("b") is None

    async def test_lru_eviction(self) -> None:
        cache = MemoryCache(max_entries=2)
        await cache.put("a", 1)
        await cache.put("b", 2)
        await cache.put("c", 3)  # Should evict "a"
        assert await cache.get("a") is None
        assert await cache.get("b") == 2
        assert await cache.get("c") == 3

    async def test_lru_access_refreshes_position(self) -> None:
        cache = MemoryCache(max_entries=2)
        await cache.put("a", 1)
        await cache.put("b", 2)
        await cache.get("a")  # Refresh "a"
        await cache.put("c", 3)  # Should evict "b" (least recently used)
        assert await cache.get("a") == 1
        assert await cache.get("b") is None
        assert await cache.get("c") == 3

    async def test_ttl_expiry(self) -> None:
        cache = MemoryCache()
        await cache.put("key1", "value1", ttl_seconds=1)
        # Manually expire the entry
        entry = cache._store["key1"]
        entry.created_at = time.time() - 2
        assert await cache.get("key1") is None

    async def test_update_existing_key(self) -> None:
        cache = MemoryCache()
        await cache.put("key1", "v1")
        await cache.put("key1", "v2")
        assert await cache.get("key1") == "v2"

    async def test_invalidate_nonexistent_key(self) -> None:
        """Invalidating a missing key should not raise."""
        cache = MemoryCache()
        await cache.invalidate("nope")  # Should not raise


class TestFakeContextCache:
    """FakeContextCache should work as a simple test double."""

    async def test_round_trip(self) -> None:
        from tests.fakes.fake_context_cache import FakeContextCache

        cache = FakeContextCache()
        await cache.put("k", "v")
        assert await cache.get("k") == "v"

    async def test_miss(self) -> None:
        from tests.fakes.fake_context_cache import FakeContextCache

        cache = FakeContextCache()
        assert await cache.get("missing") is None

    async def test_clear(self) -> None:
        from tests.fakes.fake_context_cache import FakeContextCache

        cache = FakeContextCache()
        await cache.put("k", "v")
        await cache.clear()
        assert await cache.get("k") is None
