"""Output formatter protocol — defines the contract all formatters implement."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pageindex_rag.synthesis.models import UnderwriterSummary


@runtime_checkable
class IOutputFormatter(Protocol):
    """Protocol for output formatters (PDF, HTML, JSON, etc.)."""

    def format(self, summary: UnderwriterSummary, **kwargs: Any) -> bytes:
        """Render the summary into output bytes (PDF, HTML, etc.)."""
        ...

    def format_to_file(self, summary: UnderwriterSummary, path: Path, **kwargs: Any) -> Path:
        """Render and write to a file. Returns the output path."""
        ...

    @property
    def content_type(self) -> str:
        """MIME type for the output format (e.g. 'application/pdf')."""
        ...
