"""Context factoring: multi-breakpoint prompt cache hierarchy."""

from __future__ import annotations

from scout_ai.context.factoring.breakpoint_strategy import compute_breakpoints
from scout_ai.context.factoring.layer_builder import ContextLayerBuilder

__all__ = [
    "ContextLayerBuilder",
    "compute_breakpoints",
]
