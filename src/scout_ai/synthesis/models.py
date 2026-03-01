"""Re-export shim — canonical location is ``scout_ai.domains.aps.models``.

All APS synthesis dataclasses have moved to ``domains.aps.models``.
This module re-exports them for backward compatibility::

    from scout_ai.synthesis.models import APSSummary  # still works
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
    "SynthesisSection",
    "UnderwriterSummary",
    "VitalSign",
]
