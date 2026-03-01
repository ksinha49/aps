"""Pydantic data models for scout-ai.

Core models are domain-agnostic.  Domain-specific enums and dataclasses
(e.g. ``MedicalSectionType``, ``ExtractionCategory``) live in their
respective ``domains.*`` sub-packages.  Backward-compatible re-exports
are preserved via ``__getattr__``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

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


# ── Per-run analytics ────────────────────────────────────────────────


class StageMetrics(BaseModel):
    """Metrics for a single pipeline stage (retrieval, extraction, etc.)."""

    stage: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_ms: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    success_count: int = 0
    failure_count: int = 0
    errors: list[str] = Field(default_factory=list)


class RunAnalytics(BaseModel):
    """Aggregate analytics for a single pipeline run."""

    run_id: str
    doc_id: str = ""
    started_at: datetime
    ended_at: datetime | None = None
    total_duration_ms: float = 0.0
    stages: list[StageMetrics] = Field(default_factory=list)
    total_llm_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_success: int = 0
    total_failures: int = 0
    errors: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "completed", "failed", "partial"] = "pending"

    def finalize(self) -> None:
        """Aggregate totals from stage metrics and mark completed."""
        self.ended_at = datetime.now(timezone.utc)
        if self.started_at:
            self.total_duration_ms = (self.ended_at - self.started_at).total_seconds() * 1000

        self.total_llm_calls = sum(s.llm_calls for s in self.stages)
        self.total_prompt_tokens = sum(s.prompt_tokens for s in self.stages)
        self.total_completion_tokens = sum(s.completion_tokens for s in self.stages)
        self.total_success = sum(s.success_count for s in self.stages)
        self.total_failures = sum(s.failure_count for s in self.stages)

        # Collect all stage errors
        for s in self.stages:
            self.errors.extend(s.errors)

        if self.total_failures > 0 and self.total_success > 0:
            self.status = "partial"
        elif self.total_failures > 0:
            self.status = "failed"
        else:
            self.status = "completed"


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
