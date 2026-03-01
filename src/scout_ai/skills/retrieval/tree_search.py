"""Tree search skill — single-query retrieval over document tree.

Replaces ``providers/pageindex/retrieval.py``. The agent's model
reasons over the tree structure to identify relevant nodes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from scout_ai.models import DocumentIndex, RetrievalResult
from scout_ai.providers.pageindex.tree_utils import (
    create_node_mapping,
    get_source_pages,
    tree_to_dict,
)

log = logging.getLogger(__name__)


@tool(context=True)
def tree_search(query: str, top_k: int = 5, tool_context: ToolContext = None) -> str:  # type: ignore[assignment]
    """Search the document tree index for sections relevant to a query.

    The tree structure (titles, summaries, content_types, page ranges)
    is provided to the agent's model for reasoning-based retrieval.

    Args:
        query: The search query describing what information to find.
        top_k: Maximum number of tree nodes to return.

    Returns:
        JSON with the tree structure and search prompt for the agent.
    """
    index: DocumentIndex | None = tool_context.invocation_state.get("document_index")  # type: ignore[union-attr]
    if not index:
        return json.dumps({"error": "No document_index in invocation state"})

    tree_structure = json.dumps(tree_to_dict(index.tree), indent=2)

    return json.dumps({
        "action": "tree_search",
        "query": query,
        "top_k": top_k,
        "tree_structure": tree_structure,
        "instruction": (
            "Analyze the document tree structure and identify the sections "
            "most likely to contain information relevant to the query. "
            "Consider: progress notes often contain vital signs and assessments, "
            "lab reports contain test results, imaging reports contain scan findings, "
            "discharge summaries contain treatment outcomes. "
            f"Return up to {top_k} most relevant node_ids."
        ),
    })


def resolve_search_result(
    index: DocumentIndex,
    node_ids: list[str],
    reasoning: str,
    query: str,
    top_k: int = 5,
) -> RetrievalResult:
    """Pure logic: resolve node_ids to full RetrievalResult.

    Args:
        index: The document index to search.
        node_ids: Node IDs selected by the agent.
        reasoning: Agent's reasoning for the selection.
        query: Original search query.
        top_k: Maximum results.

    Returns:
        RetrievalResult with resolved nodes and source pages.
    """
    node_map = create_node_mapping(index.tree)
    retrieved_nodes: list[dict[str, Any]] = []
    matched_nodes = []

    for nid in node_ids[:top_k]:
        node = node_map.get(nid)
        if node:
            matched_nodes.append(node)
            retrieved_nodes.append({
                "node_id": node.node_id,
                "title": node.title,
                "start_index": node.start_index,
                "end_index": node.end_index,
                "text": node.text,
            })

    source_pages = get_source_pages(matched_nodes) if matched_nodes else []

    return RetrievalResult(
        query=query,
        retrieved_nodes=retrieved_nodes,
        source_pages=source_pages,
        reasoning=reasoning,
    )
