"""S3-backed context cache — wraps IPersistenceBackend for cache semantics."""

from __future__ import annotations

import json
import time
from typing import Any

from scout_ai.context.models import CacheEntry


class S3Cache:
    """S3 context cache wrapping an existing IPersistenceBackend.

    Values are JSON-serialized with TTL metadata stored alongside.
    """

    def __init__(
        self,
        settings: Any = None,
        prefix: str = "_cache/",
    ) -> None:
        self._prefix = prefix
        self._backend = self._resolve_backend(settings)

    @staticmethod
    def _resolve_backend(settings: Any) -> Any:
        """Resolve an IPersistenceBackend from settings."""
        if settings is None:
            raise ValueError("S3Cache requires settings to resolve a persistence backend")

        # If settings has a persistence backend already, use it
        backend = getattr(settings, "_persistence_backend", None)
        if backend is not None:
            return backend

        # Otherwise create one from persistence config
        from scout_ai.persistence.file_backend import FilePersistenceBackend

        persistence_config = getattr(settings, "persistence", None)
        if persistence_config and persistence_config.backend == "s3":
            try:
                from scout_ai.persistence.s3_backend import S3PersistenceBackend

                return S3PersistenceBackend(
                    bucket=persistence_config.s3_bucket,
                    prefix=persistence_config.s3_prefix,
                )
            except ImportError:
                pass

        return FilePersistenceBackend(store_path=getattr(persistence_config, "store_path", "./indexes"))

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> Any | None:
        """Load from S3, checking TTL."""
        full_key = self._key(key)
        try:
            raw = self._backend.load(full_key)
        except KeyError:
            return None

        envelope = json.loads(raw)
        entry = CacheEntry(
            key=key,
            value=envelope["value"],
            created_at=envelope.get("created_at", 0),
            ttl_seconds=envelope.get("ttl_seconds", 0),
        )
        if entry.is_expired:
            self._backend.delete(full_key)
            return None

        return entry.value

    async def put(self, key: str, value: Any, *, ttl_seconds: int = 0) -> None:
        """Serialize value + TTL metadata to S3."""
        full_key = self._key(key)
        envelope = {
            "value": value,
            "created_at": time.time(),
            "ttl_seconds": ttl_seconds,
        }
        self._backend.save(full_key, json.dumps(envelope))

    async def invalidate(self, key: str) -> None:
        """Delete a specific cache entry from S3."""
        self._backend.delete(self._key(key))

    async def clear(self) -> None:
        """Delete all cache entries with the prefix."""
        keys = self._backend.list_keys(self._prefix)
        for k in keys:
            self._backend.delete(k)
