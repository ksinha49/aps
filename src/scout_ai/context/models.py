"""Data models for the context engineering modules."""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Literal


@dataclasses.dataclass(frozen=True)
class CompressedContext:
    """Result of a context compression operation."""

    text: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    method: str
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ContextLayer:
    """A single layer in a multi-breakpoint prompt cache hierarchy.

    Each layer represents a message segment (system, tools, document, query)
    that may or may not have a cache breakpoint at its boundary.
    """

    role: str
    content: list[dict[str, Any]]
    cache_breakpoint: bool = False
    layer_type: Literal["system", "tools", "document", "query"] = "system"


@dataclasses.dataclass
class CacheEntry:
    """Metadata wrapper for cached values with TTL tracking."""

    key: str
    value: Any
    created_at: float = dataclasses.field(default_factory=time.time)
    ttl_seconds: int = 0
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if this entry has exceeded its TTL."""
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - self.created_at) >= self.ttl_seconds
