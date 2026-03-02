"""Redis-backed context cache with async support.

Requires optional dependency: ``pip install scout-ai[redis]``
"""

from __future__ import annotations

import json
from typing import Any


class RedisCache:
    """Async Redis cache using ``redis.asyncio``.

    Requires: ``pip install scout-ai[redis]``
    """

    def __init__(self, url: str = "", max_entries: int = 1000) -> None:
        self._url = url or "redis://localhost:6379"
        self._max_entries = max_entries
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Lazy-initialize the Redis async client."""
        if self._client is not None:
            return self._client
        try:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "Redis is required for the Redis context cache backend. "
                "Install it with: pip install scout-ai[redis]"
            ) from None

        self._client = aioredis.from_url(self._url, decode_responses=True)
        return self._client

    async def get(self, key: str) -> Any | None:
        """Retrieve a cached value. Returns None on miss (Redis handles TTL)."""
        client = self._get_client()
        raw = await client.get(f"scout:cache:{key}")
        if raw is None:
            return None
        return json.loads(raw)

    async def put(self, key: str, value: Any, *, ttl_seconds: int = 0) -> None:
        """Store a value with optional TTL (Redis-native expiry)."""
        client = self._get_client()
        serialized = json.dumps(value)
        full_key = f"scout:cache:{key}"
        if ttl_seconds > 0:
            await client.setex(full_key, ttl_seconds, serialized)
        else:
            await client.set(full_key, serialized)

    async def invalidate(self, key: str) -> None:
        """Remove a specific key."""
        client = self._get_client()
        await client.delete(f"scout:cache:{key}")

    async def clear(self) -> None:
        """Remove all scout cache entries."""
        client = self._get_client()
        keys = []
        async for key in client.scan_iter(match="scout:cache:*"):
            keys.append(key)
        if keys:
            await client.delete(*keys)
