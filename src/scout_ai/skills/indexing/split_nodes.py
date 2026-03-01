"""Large node splitting skill.

Extracted from ``indexer.py:_process_large_nodes`` (lines 580-613).
Recursively subdivides nodes that exceed max_pages_per_node AND
max_tokens_per_node thresholds.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from strands import tool
from strands.types.tools import ToolContext

log = logging.getLogger(__name__)


@tool(context=True)
def split_large_nodes(  # type: ignore[assignment]
    tree_summary: list[dict[str, Any]],
    tool_context: ToolContext = None,
) -> str:
    """Identify tree nodes that are too large and need splitting.

    Scans the tree for leaf nodes that exceed both the page count
    and token count thresholds.

    Args:
        tree_summary: List of tree node dicts with title, start_index,
            end_index, children, and token_count.

    Returns:
        JSON with nodes that need splitting and their page ranges.
    """
    settings = tool_context.invocation_state.get("settings")  # type: ignore[union-attr]

    max_pages = 10
    max_tokens = 20_000
    if settings:
        max_pages = settings.indexing.max_pages_per_node
        max_tokens = settings.indexing.max_tokens_per_node

    nodes_to_split = []
    _find_large_nodes(tree_summary, nodes_to_split, max_pages, max_tokens)

    return json.dumps({
        "action": "split_large_nodes",
        "nodes_to_split": nodes_to_split,
        "max_pages_per_node": max_pages,
        "max_tokens_per_node": max_tokens,
        "instruction": (
            "For each large node, generate sub-sections by analyzing "
            "the page content within the node's page range. "
            "Return a TOC-like structure for sub-sections."
        ),
    })


def _find_large_nodes(
    nodes: list[dict[str, Any]],
    result: list[dict[str, Any]],
    max_pages: int,
    max_tokens: int,
) -> None:
    """Recursively find leaf nodes exceeding size thresholds."""
    for node in nodes:
        children = node.get("children", [])
        page_range = node.get("end_index", 0) - node.get("start_index", 0) + 1
        token_count = node.get("token_count", 0)

        if (
            page_range > max_pages
            and token_count >= max_tokens
            and not children
        ):
            result.append({
                "title": node.get("title", ""),
                "node_id": node.get("node_id", ""),
                "start_index": node.get("start_index"),
                "end_index": node.get("end_index"),
                "page_range": page_range,
                "token_count": token_count,
            })

        if children:
            _find_large_nodes(children, result, max_pages, max_tokens)
