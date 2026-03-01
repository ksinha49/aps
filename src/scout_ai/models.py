"""Pydantic data models for scout-ai."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# ── Medical section types ────────────────────────────────────────────


class MedicalSectionType(str, Enum):
    """Recognized APS medical document section types."""

    FACE_SHEET = "face_sheet"
    PROGRESS_NOTE = "progress_note"
    LAB_REPORT = "lab_report"
    IMAGING = "imaging"
    PATHOLOGY = "pathology"
    OPERATIVE_REPORT = "operative_report"
    DISCHARGE_SUMMARY = "discharge_summary"
    CONSULTATION = "consultation"
    MEDICATION_LIST = "medication_list"
    VITAL_SIGNS = "vital_signs"
    NURSING_NOTE = "nursing_note"
    THERAPY_NOTE = "therapy_note"
    MENTAL_HEALTH = "mental_health"
    DENTAL = "dental"
    UNKNOWN = "unknown"


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
    content_type: MedicalSectionType = MedicalSectionType.UNKNOWN
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


# ── Extraction categories ────────────────────────────────────────────


class ExtractionCategory(str, Enum):
    """16 APS extraction categories matching the schema sections."""

    DEMOGRAPHICS = "demographics"
    EMPLOYMENT = "employment"
    MEDICAL_HISTORY = "medical_history"
    CURRENT_MEDICATIONS = "current_medications"
    ALLERGIES = "allergies"
    VITAL_SIGNS = "vital_signs"
    PHYSICAL_EXAM = "physical_exam"
    LAB_RESULTS = "lab_results"
    IMAGING_RESULTS = "imaging_results"
    DIAGNOSES = "diagnoses"
    PROCEDURES = "procedures"
    MENTAL_HEALTH = "mental_health"
    FUNCTIONAL_CAPACITY = "functional_capacity"
    TREATMENT_PLAN = "treatment_plan"
    PROGNOSIS = "prognosis"
    PHYSICIAN_OPINION = "physician_opinion"


# ── Extraction question / result models ──────────────────────────────


class ExtractionQuestion(BaseModel):
    """A single extraction question with tier assignment."""

    question_id: str
    category: ExtractionCategory
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

    category: ExtractionCategory
    retrieval: RetrievalResult
    extractions: list[ExtractionResult] = Field(default_factory=list)
