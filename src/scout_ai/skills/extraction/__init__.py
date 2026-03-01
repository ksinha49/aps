"""Extraction skills: batch and individual answer extraction."""

from __future__ import annotations

from scout_ai.skills.extraction.extract_batch import extract_batch
from scout_ai.skills.extraction.extract_individual import extract_individual

__all__ = ["extract_batch", "extract_individual"]
