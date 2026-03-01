"""Exception hierarchy for scout-ai.

Re-exports the original exceptions and adds new ones for the Strands-based
architecture (checkpoint, persistence, hook errors).
"""

from __future__ import annotations

from scout_ai.exceptions import (
    ExtractionError,
    IndexBuildError,
    LLMClientError,
    RetrievalError,
    ScoutError,
    TokenizerError,
)


class CheckpointError(ScoutError):
    """Raised when checkpoint save/load fails."""


class PersistenceError(ScoutError):
    """Raised when a persistence backend operation fails."""


class HookError(ScoutError):
    """Raised when a lifecycle hook encounters an error."""


__all__ = [
    "ScoutError",
    "IndexBuildError",
    "RetrievalError",
    "ExtractionError",
    "LLMClientError",
    "TokenizerError",
    "CheckpointError",
    "PersistenceError",
    "HookError",
]
