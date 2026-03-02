"""Breakpoint placement logic for multi-layer prompt caching.

Anthropic supports up to 4 cache breakpoints per request. This module
decides which layer boundaries get ``cache_control`` markers to maximize
cache reuse while respecting the breakpoint budget.

Priority order (highest cache value first):
1. system  — rarely changes, shared across all calls
2. document — changes per document but shared across questions
3. tools   — optional, stable across calls
4. query   — changes every call, never cached
"""

from __future__ import annotations

from scout_ai.context.models import ContextLayer

# Priority: layers that change least should be cached first.
_LAYER_PRIORITY = {"system": 0, "tools": 1, "document": 2, "query": 3}


def compute_breakpoints(
    layers: list[ContextLayer],
    max_breakpoints: int = 4,
) -> list[ContextLayer]:
    """Assign cache breakpoints to the highest-value layer boundaries.

    Mutates ``cache_breakpoint`` on the input layers and returns them.
    The query layer is never assigned a breakpoint (it changes every call).

    Args:
        layers: Ordered list of context layers.
        max_breakpoints: Maximum breakpoints allowed (Anthropic limit is 4).

    Returns:
        The same list with ``cache_breakpoint`` flags set.
    """
    # Reset all breakpoints first
    for layer in layers:
        layer.cache_breakpoint = False

    # Sort candidates by priority (lowest = most valuable to cache)
    candidates = [
        layer for layer in layers if layer.layer_type != "query"
    ]
    candidates.sort(key=lambda ly: _LAYER_PRIORITY.get(ly.layer_type, 99))

    # Assign breakpoints up to the limit
    for layer in candidates[:max_breakpoints]:
        layer.cache_breakpoint = True

    return layers
