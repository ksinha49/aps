"""Node enrichment skill — summaries, classification, description.

Extracted from ``indexer.py:_generate_summaries``, ``_classify_nodes``,
``_generate_doc_description``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from strands import tool
from strands.types.tools import ToolContext

log = logging.getLogger(__name__)


@tool(context=True)
def enrich_nodes(  # type: ignore[assignment]
    tree_summary: list[dict[str, Any]],
    enable_summaries: bool = True,
    enable_classification: bool = True,
    enable_description: bool = False,
    tool_context: ToolContext = None,
) -> str:
    """Enrich tree nodes with summaries, section classification, and doc description.

    Args:
        tree_summary: Flattened list of node dicts with title, text, node_id.
        enable_summaries: Generate 2-3 sentence summaries for each node.
        enable_classification: Classify nodes by domain section type.
        enable_description: Generate a one-sentence document description.

    Returns:
        JSON with enrichment instructions and nodes to process.
    """
    from scout_ai.core.config import IndexingConfig

    settings = tool_context.invocation_state.get("settings")  # type: ignore[union-attr]
    idx_cfg = settings.indexing if settings else IndexingConfig()
    summary_max_chars = idx_cfg.summary_max_chars
    classification_max_chars = idx_cfg.classification_max_chars

    nodes_for_summary = []
    nodes_for_classification = []

    for node in tree_summary:
        text = node.get("text", "")
        if enable_summaries and text:
            nodes_for_summary.append({
                "node_id": node.get("node_id", ""),
                "title": node.get("title", ""),
                "text_preview": text[:summary_max_chars],
                "text_length": len(text),
            })

        if enable_classification:
            nodes_for_classification.append({
                "node_id": node.get("node_id", ""),
                "title": node.get("title", ""),
                "content_preview": text[:classification_max_chars],
            })

    result: dict[str, Any] = {
        "action": "enrich_nodes",
        "enable_summaries": enable_summaries,
        "enable_classification": enable_classification,
        "enable_description": enable_description,
    }

    if enable_summaries:
        result["nodes_for_summary"] = nodes_for_summary
        result["summary_instruction"] = (
            "For each node, generate a concise 2-3 sentence summary "
            "focusing on clinically relevant information: diagnoses, "
            "findings, measurements, dates."
        )

    if enable_classification:
        result["nodes_for_classification"] = nodes_for_classification
        # Resolve section types from domain registry
        section_types = _get_section_types(tool_context)
        type_list = ", ".join(section_types) if section_types else "unknown"
        result["classification_instruction"] = (
            f"For each node, classify into one of: {type_list}."
        )

    if enable_description:
        result["description_instruction"] = (
            "Generate a one-sentence description for the entire document "
            "based on the tree structure and node summaries."
        )

    return json.dumps(result)


def _get_section_types(tool_context: ToolContext) -> list[str]:
    """Resolve section types from invocation state or domain registry."""
    types = tool_context.invocation_state.get("section_types")  # type: ignore[union-attr]
    if types:
        return types

    settings = tool_context.invocation_state.get("settings")  # type: ignore[union-attr]
    domain_name = "aps"
    if settings and hasattr(settings, "domain"):
        domain_name = settings.domain

    try:
        from scout_ai.domains.registry import get_registry

        registry = get_registry()
        domain_config = registry.get(domain_name)
        return domain_config.section_types
    except (KeyError, ImportError):
        return ["unknown"]
