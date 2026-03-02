"""Context engineering modules: compression, factoring, prefix stabilization, and caching.

All modules are disabled by default and opt-in via ``SCOUT_*`` env vars.
"""

from __future__ import annotations

from scout_ai.context.models import CacheEntry, CompressedContext, ContextLayer
from scout_ai.context.protocols import IContextCache, IContextCompressor

__all__ = [
    "IContextCompressor",
    "IContextCache",
    "CompressedContext",
    "ContextLayer",
    "CacheEntry",
    "create_compressor",
    "create_context_cache",
]


def create_compressor(settings: object | None = None) -> IContextCompressor:
    """Factory: create a context compressor from settings.

    Lazy import to avoid circular deps at module load time.
    """
    from scout_ai.context.compression import create_compressor as _factory

    return _factory(settings)


def create_context_cache(settings: object | None = None) -> IContextCache:
    """Factory: create a context cache from settings.

    Lazy import to avoid circular deps at module load time.
    """
    from scout_ai.context.cache import create_context_cache as _factory

    return _factory(settings)
