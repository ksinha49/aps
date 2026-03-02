"""Context/extraction result caching: factory + backend implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scout_ai.context.cache.memory import MemoryCache
from scout_ai.context.protocols import IContextCache

if TYPE_CHECKING:
    from scout_ai.core.config import ContextCacheConfig

__all__ = [
    "create_context_cache",
    "MemoryCache",
]


def create_context_cache(settings: object | None = None) -> IContextCache:
    """Create a context cache from settings.

    Args:
        settings: An ``AppSettings`` or ``ContextCacheConfig`` instance.
            If None, returns MemoryCache with defaults.
    """
    config: ContextCacheConfig | None = None

    if settings is not None:
        config = getattr(settings, "context_cache", None)
        if config is None and hasattr(settings, "backend"):
            config = settings  # type: ignore[assignment]

    if config is None or not config.enabled:
        return MemoryCache()

    backend = config.backend
    if backend == "memory":
        return MemoryCache(max_entries=config.max_entries)
    elif backend == "s3":
        from scout_ai.context.cache.s3 import S3Cache

        return S3Cache(settings=settings)
    elif backend == "redis":
        from scout_ai.context.cache.redis import RedisCache

        return RedisCache(url=config.redis_url, max_entries=config.max_entries)
    else:
        raise ValueError(f"Unknown context cache backend: {backend!r}")
