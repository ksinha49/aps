"""Output formatters for rendering UnderwriterSummary to various formats.

Usage::

    from pageindex_rag.formatters import PDFFormatter, JSONFormatter

    pdf = PDFFormatter()
    pdf_bytes = pdf.format(summary)

    js = JSONFormatter()
    json_bytes = js.format(summary)
"""

from __future__ import annotations

from pageindex_rag.formatters.json_formatter import JSONFormatter
from pageindex_rag.formatters.protocols import IOutputFormatter

__all__ = [
    "IOutputFormatter",
    "JSONFormatter",
    "PDFFormatter",
]


def __getattr__(name: str) -> type:
    """Lazy-load PDFFormatter so reportlab is only imported when needed."""
    if name == "PDFFormatter":
        from pageindex_rag.formatters.pdf_formatter import PDFFormatter

        return PDFFormatter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
