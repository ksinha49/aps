"""Pydantic data models for scout-ai.

Core models are domain-agnostic.  Domain-specific enums and dataclasses
(e.g. ``MedicalSectionType``, ``ExtractionCategory``) live in their
respective ``domains.*`` sub-packages.  Backward-compatible re-exports
are preserved via ``__getattr__``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

# ── Document / page models ───────────────────────────────────────────


class PageContent(BaseModel):
    """A single pre-OCR'd page."""

    page_number: int
    text: str
    token_count: Optional[int] = None


class TreeNode(BaseModel):
    """A node in the hierarchical document tree."""

    node_id: str = ""
    title: str
    start_index: int
    end_index: int
    text: str = ""
    summary: str = ""
    content_type: str = "unknown"
    children: list[TreeNode] = Field(default_factory=list)

    # Metadata carried during construction (not persisted in final output)
    structure: Optional[str] = None
    physical_index: Optional[int] = None
    appear_start: Optional[str] = None


class DocumentIndex(BaseModel):
    """Persisted tree index for a document."""

    doc_id: str
    doc_name: str
    doc_description: str = ""
    total_pages: int
    tree: list[TreeNode]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Extraction question / result models ──────────────────────────────


class ExtractionQuestion(BaseModel):
    """A single extraction question with tier assignment."""

    question_id: str
    category: str
    question_text: str
    tier: int = Field(default=1, ge=1, le=3)
    expected_type: str = "text"


class RetrievalResult(BaseModel):
    """Result of a tree search for a query."""

    query: str
    retrieved_nodes: list[dict[str, Any]] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    reasoning: str = ""


class Citation(BaseModel):
    """A single verifiable reference to source document text."""

    page_number: int
    section_title: str = ""
    section_type: str = ""  # content_type value (e.g. "lab_report")
    verbatim_quote: str  # exact text from the source


class ExtractionResult(BaseModel):
    """Answer to a single extraction question."""

    question_id: str
    answer: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)
    # Kept for backward compat — derived from citations when available
    source_pages: list[int] = Field(default_factory=list)
    evidence_text: str = ""


class BatchExtractionResult(BaseModel):
    """Results for an entire extraction category."""

    category: str
    retrieval: RetrievalResult
    extractions: list[ExtractionResult] = Field(default_factory=list)


# ── Backward compatibility shim ──────────────────────────────────────
# ``from scout_ai.models import MedicalSectionType`` still works.

_COMPAT_NAMES = {"MedicalSectionType", "ExtractionCategory"}


def __getattr__(name: str) -> Any:
    if name in _COMPAT_NAMES:
        from scout_ai.domains.aps.models import (
            ExtractionCategory as _EC,
        )
        from scout_ai.domains.aps.models import (
            MedicalSectionType as _MST,
        )

        _map = {"MedicalSectionType": _MST, "ExtractionCategory": _EC}
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
