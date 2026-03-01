"""In-memory persistence backend — dict-backed, ideal for tests."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class MemoryPersistenceBackend:
    """Stores data in a plain dict — nothing touches disk."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def save(self, key: str, data: str) -> None:
        self._store[key] = data
        log.debug(f"Saved {key} to memory store")

    def load(self, key: str) -> str:
        if key not in self._store:
            raise KeyError(f"Not found in memory store: {key}")
        return self._store[key]

    def exists(self, key: str) -> bool:
        return key in self._store

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def list_keys(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self._store if k.startswith(prefix))
