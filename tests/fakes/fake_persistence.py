"""In-memory persistence backend for testing."""

from __future__ import annotations

from typing import Any


class FakePersistenceBackend:
    """Dict-backed storage for tests — no filesystem or S3 needed."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def save(self, key: str, data: Any) -> None:
        self._store[key] = data

    def load(self, key: str) -> Any:
        if key not in self._store:
            raise KeyError(f"Not found: {key}")
        return self._store[key]

    def exists(self, key: str) -> bool:
        return key in self._store

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def list_keys(self, prefix: str = "") -> list[str]:
        return [k for k in self._store if k.startswith(prefix)]
