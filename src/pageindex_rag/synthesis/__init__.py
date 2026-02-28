"""Synthesis pipeline: aggregates extraction results into underwriter summaries."""

from __future__ import annotations

from pageindex_rag.synthesis.models import SynthesisSection, UnderwriterSummary
from pageindex_rag.synthesis.pipeline import SynthesisPipeline

__all__ = [
    "SynthesisPipeline",
    "SynthesisSection",
    "UnderwriterSummary",
]
