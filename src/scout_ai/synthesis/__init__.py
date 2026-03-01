"""Synthesis pipeline: aggregates extraction results into underwriter summaries."""

from __future__ import annotations

from scout_ai.synthesis.models import SynthesisSection, UnderwriterSummary
from scout_ai.synthesis.pipeline import SynthesisPipeline

__all__ = [
    "SynthesisPipeline",
    "SynthesisSection",
    "UnderwriterSummary",
]
