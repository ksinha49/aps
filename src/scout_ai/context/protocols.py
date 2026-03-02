"""Context engineering protocols — contracts for compression and caching backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from scout_ai.context.models import CompressedContext


@runtime_checkable
class IContextCompressor(Protocol):
    """Protocol for context compression backends.

    All compressors are synchronous — compression is CPU-bound,
    not I/O-bound, so async adds no benefit.
    """

    def compress(self, text: str, *, target_ratio: float = 0.5) -> CompressedContext:
        """Compress text, targeting the given ratio (0.0–1.0).

        Args:
            text: Raw context text to compress.
            target_ratio: Desired compressed/original ratio. Lower = more aggressive.

        Returns:
            CompressedContext with compressed text and metadata.
        """
        ...


@runtime_checkable
class IContextCache(Protocol):
    """Protocol for async context/extraction result caching backends."""

    async def get(self, key: str) -> Any | None:
        """Retrieve a cached value by key. Returns None on miss."""
        ...

    async def put(self, key: str, value: Any, *, ttl_seconds: int = 0) -> None:
        """Store a value under the given key with optional TTL.

        Args:
            key: Cache key.
            value: JSON-serializable value.
            ttl_seconds: Time-to-live in seconds. 0 = use backend default.
        """
        ...

    async def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache."""
        ...

    async def clear(self) -> None:
        """Remove all entries from the cache."""
        ...
