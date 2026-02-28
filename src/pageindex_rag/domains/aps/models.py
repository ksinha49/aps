"""APS domain enums — re-exported from the canonical location in models.py.

The enums are defined in ``pageindex_rag.models`` (which owns the Pydantic
models that reference them).  This module re-exports them so that
domain-specific code can import from ``domains.aps.models``.
"""

from __future__ import annotations

from pageindex_rag.models import ExtractionCategory, MedicalSectionType

__all__ = ["ExtractionCategory", "MedicalSectionType"]
