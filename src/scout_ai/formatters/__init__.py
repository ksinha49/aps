"""Output formatters for rendering UnderwriterSummary to various formats.

Usage::

    from scout_ai.formatters import PDFFormatter, JSONFormatter

    pdf = PDFFormatter()
    pdf_bytes = pdf.format(summary)

    js = JSONFormatter()
    json_bytes = js.format(summary)
"""

from __future__ import annotations

from typing import Any

from scout_ai.formatters.json_formatter import JSONFormatter
from scout_ai.formatters.protocols import IOutputFormatter

__all__ = [
    "IOutputFormatter",
    "JSONFormatter",
    "PDFFormatter",
]


def __getattr__(name: str) -> Any:
    """Lazy-load PDFFormatter so reportlab is only imported when needed."""
    if name == "PDFFormatter":
        from scout_ai.formatters.pdf_formatter import PDFFormatter

        return PDFFormatter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
