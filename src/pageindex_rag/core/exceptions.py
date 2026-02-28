"""Exception hierarchy for pageindex-rag.

Re-exports the original exceptions and adds new ones for the Strands-based
architecture (checkpoint, persistence, hook errors).
"""

from __future__ import annotations

from pageindex_rag.exceptions import (
    ExtractionError,
    IndexBuildError,
    LLMClientError,
    PageIndexError,
    RetrievalError,
    TokenizerError,
)


class CheckpointError(PageIndexError):
    """Raised when checkpoint save/load fails."""


class PersistenceError(PageIndexError):
    """Raised when a persistence backend operation fails."""


class HookError(PageIndexError):
    """Raised when a lifecycle hook encounters an error."""


__all__ = [
    "PageIndexError",
    "IndexBuildError",
    "RetrievalError",
    "ExtractionError",
    "LLMClientError",
    "TokenizerError",
    "CheckpointError",
    "PersistenceError",
    "HookError",
]
