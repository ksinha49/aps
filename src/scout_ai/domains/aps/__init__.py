"""APS (Attending Physician Statement) medical domain module."""

from __future__ import annotations

from scout_ai.domains.aps.categories import CATEGORY_DESCRIPTIONS
from scout_ai.domains.aps.classifier import MedicalSectionClassifier
from scout_ai.domains.aps.models import ExtractionCategory, MedicalSectionType
from scout_ai.domains.aps.section_patterns import SECTION_PATTERNS

__all__ = [
    "CATEGORY_DESCRIPTIONS",
    "ExtractionCategory",
    "MedicalSectionClassifier",
    "MedicalSectionType",
    "SECTION_PATTERNS",
]
