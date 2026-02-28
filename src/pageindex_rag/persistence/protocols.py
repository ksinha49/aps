"""Persistence backend protocol — defines the contract all backends implement."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IPersistenceBackend(Protocol):
    """Protocol for persistence backends (file, S3, memory, etc.)."""

    def save(self, key: str, data: str) -> None:
        """Save serialized data under the given key."""
        ...

    def load(self, key: str) -> str:
        """Load serialized data by key. Raises KeyError if not found."""
        ...

    def exists(self, key: str) -> bool:
        """Check if a key exists in the store."""
        ...

    def delete(self, key: str) -> None:
        """Delete data by key (no-op if not found)."""
        ...

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys matching the optional prefix."""
        ...
