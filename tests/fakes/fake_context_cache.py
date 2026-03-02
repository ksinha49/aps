"""In-memory context cache fake for testing."""

from __future__ import annotations

from typing import Any


class FakeContextCache:
    """Dict-backed IContextCache for tests — no async overhead."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    async def get(self, key: str) -> Any | None:
        return self._store.get(key)

    async def put(self, key: str, value: Any, *, ttl_seconds: int = 0) -> None:
        self._store[key] = value

    async def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    async def clear(self) -> None:
        self._store.clear()
