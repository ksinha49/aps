"""APS module — re-exports from domains.aps for backward compatibility."""

from __future__ import annotations

from pageindex_rag.domains.aps.categories import CATEGORY_DESCRIPTIONS
from pageindex_rag.domains.aps.classifier import MedicalSectionClassifier
from pageindex_rag.domains.aps.models import ExtractionCategory, MedicalSectionType
from pageindex_rag.domains.aps.section_patterns import SECTION_PATTERNS

__all__ = [
    "CATEGORY_DESCRIPTIONS",
    "ExtractionCategory",
    "MedicalSectionClassifier",
    "MedicalSectionType",
    "SECTION_PATTERNS",
]
