"""Data models for the synthesis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SynthesisSection:
    """A single section of the underwriter summary."""

    title: str
    content: str
    source_categories: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)


@dataclass
class UnderwriterSummary:
    """Structured underwriter summary produced from extraction results."""

    document_id: str
    patient_demographics: str
    sections: list[SynthesisSection] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    overall_assessment: str = ""
    total_questions_answered: int = 0
    high_confidence_count: int = 0
    generated_at: str = ""
