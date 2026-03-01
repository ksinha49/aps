"""Workers' Compensation domain module.

Provides extraction categories, section types, and prompts for
workers' compensation claim document processing.
"""

from __future__ import annotations

from scout_ai.domains.workers_comp.categories import CATEGORY_DESCRIPTIONS
from scout_ai.domains.workers_comp.models import (
    ExtractionCategory,
    SectionType,
)

__all__ = [
    "CATEGORY_DESCRIPTIONS",
    "ExtractionCategory",
    "SectionType",
]
