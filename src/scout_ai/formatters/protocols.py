"""Output formatter protocol — defines the contract all formatters implement.

This protocol is domain-agnostic.  The ``summary`` parameter accepts ``Any``
so that domain-specific formatter implementations can type-narrow as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IOutputFormatter(Protocol):
    """Protocol for output formatters (PDF, HTML, JSON, etc.).

    Implementations dispatch on the summary type to render the
    appropriate layout.
    """

    def format(self, summary: Any, **kwargs: Any) -> bytes:
        """Render the summary into output bytes (PDF, HTML, etc.)."""
        ...

    def format_to_file(self, summary: Any, path: Path, **kwargs: Any) -> Path:
        """Render and write to a file. Returns the output path."""
        ...

    @property
    def content_type(self) -> str:
        """MIME type for the output format (e.g. 'application/pdf')."""
        ...


# Backward-compat alias
SummaryInput = Any

__all__ = ["IOutputFormatter", "SummaryInput"]
