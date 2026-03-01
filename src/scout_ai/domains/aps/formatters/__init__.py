"""APS output formatters: PDF, JSON rendering for APSSummary / UnderwriterSummary."""

from __future__ import annotations

from typing import Any

from scout_ai.domains.aps.formatters.json_formatter import JSONFormatter

__all__ = [
    "JSONFormatter",
    "PDFFormatter",
]


def __getattr__(name: str) -> Any:
    """Lazy-load PDFFormatter so reportlab is only imported when needed."""
    if name == "PDFFormatter":
        from scout_ai.domains.aps.formatters.pdf_formatter import PDFFormatter

        return PDFFormatter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
