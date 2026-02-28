"""Extraction skills: batch and individual answer extraction."""

from __future__ import annotations

from pageindex_rag.skills.extraction.extract_batch import extract_batch
from pageindex_rag.skills.extraction.extract_individual import extract_individual

__all__ = ["extract_batch", "extract_individual"]
