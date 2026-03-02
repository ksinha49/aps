"""In-memory LRU cache with TTL support and async safety."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any

from scout_ai.context.models import CacheEntry


class MemoryCache:
    """OrderedDict-based LRU cache with TTL expiry.

    Thread-safe via ``asyncio.Lock`` — safe for concurrent coroutine access.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self._max_entries = max_entries
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """Retrieve a value by key. Returns None on miss or TTL expiry."""
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.is_expired:
                del self._store[key]
                return None
            entry.hit_count += 1
            # Move to end (most recently used)
            self._store.move_to_end(key)
            return entry.value

    async def put(self, key: str, value: Any, *, ttl_seconds: int = 0) -> None:
        """Store a value with optional TTL. Evicts LRU entries if at capacity."""
        async with self._lock:
            # Remove existing entry if present (to update position)
            if key in self._store:
                del self._store[key]

            self._store[key] = CacheEntry(
                key=key,
                value=value,
                ttl_seconds=ttl_seconds,
            )
            self._store.move_to_end(key)

            # Evict oldest entries if over capacity
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    async def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache."""
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        """Remove all entries."""
        async with self._lock:
            self._store.clear()
