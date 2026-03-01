"""Synthesis pipeline: aggregates extraction results into underwriter summaries.

Re-exports from canonical locations for backward compatibility.
"""

from __future__ import annotations

from scout_ai.domains.aps.models import (
    Allergy,
    APSSection,
    APSSummary,
    CitationRef,
    Condition,
    Encounter,
    Finding,
    ImagingResult,
    LabResult,
    Medication,
    PatientDemographics,
    RedFlag,
    RiskClassification,
    SurgicalHistory,
    SynthesisSection,
    UnderwriterSummary,
    VitalSign,
)
from scout_ai.domains.aps.synthesis.pipeline import SynthesisPipeline

__all__ = [
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
    "SynthesisPipeline",
    "SynthesisSection",
    "UnderwriterSummary",
    "VitalSign",
]
