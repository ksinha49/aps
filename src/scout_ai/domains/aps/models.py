"""APS domain models — enums, dataclasses, and rich types.

This is the **canonical** location for all APS-specific data structures.
Other modules (``synthesis``, ``validation``, ``formatters``) import from here.

Backward-compatible re-exports exist in:
- ``scout_ai.models`` (via ``__getattr__``)
- ``scout_ai.synthesis.models`` (thin shim)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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


# ── Legacy synthesis models ──────────────────────────────────────────


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


# ── APS Schema v1.0.0 models ────────────────────────────────────────


@dataclass
class CitationRef:
    """Inline citation reference linking a finding to its source page."""

    page_number: int
    date: str = ""
    source_type: str = ""
    section_title: str = ""
    verbatim_quote: str = ""

    def display(self) -> str:
        """Human-readable citation string, e.g. 'p.42, 03/2024 - Progress Note'."""
        parts = [f"p.{self.page_number}"]
        if self.date:
            parts.append(self.date)
        if self.source_type:
            parts.append(f"- {self.source_type}")
        return ", ".join(parts) if len(parts) == 1 else f"{parts[0]}, {', '.join(parts[1:])}"


@dataclass
class Finding:
    """A clinical finding with severity and source citations."""

    text: str
    severity: str = "INFORMATIONAL"
    citations: list[CitationRef] = field(default_factory=list)


@dataclass
class PatientDemographics:
    """Structured patient demographics with fallback to raw text."""

    full_name: str = ""
    date_of_birth: str = ""
    age: str = ""
    gender: str = ""
    ssn_last4: str = ""
    address: str = ""
    phone: str = ""
    insurance_id: str = ""
    employer: str = ""
    occupation: str = ""
    raw_text: str = ""

    def summary_text(self) -> str:
        """Single-line demographics summary for display."""
        if self.full_name:
            parts = [self.full_name]
            if self.date_of_birth:
                parts.append(f"DOB {self.date_of_birth}")
            if self.age:
                parts.append(f"Age {self.age}")
            if self.gender:
                parts.append(self.gender)
            return ", ".join(parts)
        return self.raw_text or "Demographics not available"


@dataclass
class Condition:
    """A medical condition with ICD-10 code and status tracking."""

    name: str
    icd10_code: str = ""
    onset_date: str = ""
    status: str = ""
    severity: str = ""
    citations: list[CitationRef] = field(default_factory=list)


@dataclass
class Medication:
    """A medication entry with dosage details."""

    name: str
    dose: str = ""
    frequency: str = ""
    route: str = ""
    prescriber: str = ""
    start_date: str = ""
    citations: list[CitationRef] = field(default_factory=list)


@dataclass
class LabResult:
    """A single laboratory test result."""

    test_name: str
    value: str = ""
    unit: str = ""
    reference_range: str = ""
    flag: str = ""  # H, L, C, or "" for normal
    date: str = ""
    citations: list[CitationRef] = field(default_factory=list)


@dataclass
class ImagingResult:
    """Imaging or diagnostic study result."""

    modality: str
    body_part: str = ""
    finding: str = ""
    impression: str = ""
    date: str = ""
    citations: list[CitationRef] = field(default_factory=list)


@dataclass
class Encounter:
    """A clinical encounter entry."""

    date: str
    provider: str = ""
    encounter_type: str = ""
    summary: str = ""
    citations: list[CitationRef] = field(default_factory=list)


@dataclass
class VitalSign:
    """A single vital sign measurement."""

    name: str
    value: str
    date: str = ""
    flag: str = ""
    citations: list[CitationRef] = field(default_factory=list)


@dataclass
class Allergy:
    """A recorded allergy."""

    allergen: str
    reaction: str = ""
    severity: str = ""
    citations: list[CitationRef] = field(default_factory=list)


@dataclass
class SurgicalHistory:
    """A surgical procedure record."""

    procedure: str
    date: str = ""
    outcome: str = ""
    complications: str = ""
    citations: list[CitationRef] = field(default_factory=list)


@dataclass
class RiskClassification:
    """Underwriting risk classification."""

    tier: str = ""
    table_rating: str = ""
    debit_credits: str = ""
    rationale: str = ""
    confidence: float = 0.0


@dataclass
class RedFlag:
    """A red flag or alert for underwriter attention."""

    description: str
    severity: str = "MODERATE"
    category: str = ""
    citations: list[CitationRef] = field(default_factory=list)


@dataclass
class APSSection:
    """A richly-typed section of the APS summary."""

    section_key: str
    section_number: str = ""
    title: str = ""
    content: str = ""
    source_categories: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    conditions: list[Condition] = field(default_factory=list)
    medications: list[Medication] = field(default_factory=list)
    lab_results: list[LabResult] = field(default_factory=list)
    imaging_results: list[ImagingResult] = field(default_factory=list)
    encounters: list[Encounter] = field(default_factory=list)
    vital_signs: list[VitalSign] = field(default_factory=list)
    allergies: list[Allergy] = field(default_factory=list)
    surgical_history: list[SurgicalHistory] = field(default_factory=list)


@dataclass
class APSSummary:
    """Full APS Schema v1.0.0 summary with richly-typed sections.

    Use ``to_underwriter_summary()`` for backward compatibility with
    the legacy ``UnderwriterSummary`` API surface.
    """

    document_id: str
    demographics: PatientDemographics = field(default_factory=PatientDemographics)
    sections: list[APSSection] = field(default_factory=list)
    risk_classification: RiskClassification = field(default_factory=RiskClassification)
    risk_factors: list[str] = field(default_factory=list)
    red_flags: list[RedFlag] = field(default_factory=list)
    overall_assessment: str = ""
    citation_index: dict[int, list[CitationRef]] = field(default_factory=dict)
    total_questions_answered: int = 0
    high_confidence_count: int = 0
    generated_at: str = ""

    def to_underwriter_summary(self) -> UnderwriterSummary:
        """Convert to legacy ``UnderwriterSummary`` for backward compatibility."""
        legacy_sections = [
            SynthesisSection(
                title=s.title or s.section_key,
                content=s.content,
                source_categories=s.source_categories,
                key_findings=[f.text for f in s.findings],
            )
            for s in self.sections
        ]
        return UnderwriterSummary(
            document_id=self.document_id,
            patient_demographics=self.demographics.summary_text(),
            sections=legacy_sections,
            risk_factors=self.risk_factors,
            overall_assessment=self.overall_assessment,
            total_questions_answered=self.total_questions_answered,
            high_confidence_count=self.high_confidence_count,
            generated_at=self.generated_at,
        )


# ── Underwriting-specific models ────────────────────────────────────


@dataclass
class YNCondition:
    """Y/N condition entry for the underwriting template."""

    condition_name: str
    time_qualifier: str = ""
    answer_yn: str = ""  # "Y", "N", or "Unknown"
    detail: str = ""
    citations: list[CitationRef] = field(default_factory=list)


@dataclass
class UnderwritingAPSSummary(APSSummary):
    """APS summary extended with underwriting template sections."""

    policy_number: str = ""
    aps_date_range: str = ""
    total_document_pages: int = 0
    yn_conditions: list[YNCondition] = field(default_factory=list)
    morbidity_concerns: str = ""
    morbidity_citations: list[CitationRef] = field(default_factory=list)
    mortality_concerns: str = ""
    mortality_citations: list[CitationRef] = field(default_factory=list)
    residence_travel: str = ""
    residence_citations: list[CitationRef] = field(default_factory=list)


__all__ = [
    "MedicalSectionType",
    "ExtractionCategory",
    "SynthesisSection",
    "UnderwriterSummary",
    "CitationRef",
    "Finding",
    "PatientDemographics",
    "Condition",
    "Medication",
    "LabResult",
    "ImagingResult",
    "Encounter",
    "VitalSign",
    "Allergy",
    "SurgicalHistory",
    "RiskClassification",
    "RedFlag",
    "APSSection",
    "APSSummary",
    "YNCondition",
    "UnderwritingAPSSummary",
]
