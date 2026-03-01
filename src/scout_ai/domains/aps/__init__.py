"""APS (Attending Physician Statement) medical domain module.

This is the canonical home for all APS-specific logic:
- Enums: ``MedicalSectionType``, ``ExtractionCategory``
- Models: ``APSSummary``, ``UnderwriterSummary``, and all rich dataclasses
- Categories: ``CATEGORY_DESCRIPTIONS``
- Classification: ``MedicalSectionClassifier``
- Regex patterns: ``SECTION_PATTERNS``
"""

from __future__ import annotations

from scout_ai.domains.aps.categories import CATEGORY_DESCRIPTIONS
from scout_ai.domains.aps.classifier import MedicalSectionClassifier
from scout_ai.domains.aps.models import (
    Allergy,
    APSSection,
    APSSummary,
    CitationRef,
    Condition,
    Encounter,
    ExtractionCategory,
    Finding,
    ImagingResult,
    LabResult,
    MedicalSectionType,
    Medication,
    PatientDemographics,
    RedFlag,
    RiskClassification,
    SurgicalHistory,
    SynthesisSection,
    UnderwriterSummary,
    VitalSign,
)
from scout_ai.domains.aps.section_patterns import SECTION_PATTERNS

__all__ = [
    # Enums
    "ExtractionCategory",
    "MedicalSectionType",
    # Models
    "Allergy",
    "APSSection",
    "APSSummary",
    "CitationRef",
    "Condition",
    "Encounter",
    "Finding",
    "ImagingResult",
    "LabResult",
    "Medication",
    "PatientDemographics",
    "RedFlag",
    "RiskClassification",
    "SurgicalHistory",
    "SynthesisSection",
    "UnderwriterSummary",
    "VitalSign",
    # Domain data
    "CATEGORY_DESCRIPTIONS",
    "MedicalSectionClassifier",
    "SECTION_PATTERNS",
]
