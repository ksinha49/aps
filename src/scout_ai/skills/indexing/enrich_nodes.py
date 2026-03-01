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
    """Enrich tree nodes with summaries, medical classification, and doc description.

    Args:
        tree_summary: Flattened list of node dicts with title, text, node_id.
        enable_summaries: Generate 2-3 sentence summaries for each node.
        enable_classification: Classify nodes by medical section type.
        enable_description: Generate a one-sentence document description.

    Returns:
        JSON with enrichment instructions and nodes to process.
    """
    nodes_for_summary = []
    nodes_for_classification = []

    for node in tree_summary:
        text = node.get("text", "")
        if enable_summaries and text:
            nodes_for_summary.append({
                "node_id": node.get("node_id", ""),
                "title": node.get("title", ""),
                "text_preview": text[:4000],
                "text_length": len(text),
            })

        if enable_classification:
            nodes_for_classification.append({
                "node_id": node.get("node_id", ""),
                "title": node.get("title", ""),
                "content_preview": text[:500],
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
        result["classification_instruction"] = (
            "For each node, classify into one of: face_sheet, progress_note, "
            "lab_report, imaging, pathology, operative_report, discharge_summary, "
            "consultation, medication_list, vital_signs, nursing_note, "
            "therapy_note, mental_health, dental, unknown."
        )

    if enable_description:
        result["description_instruction"] = (
            "Generate a one-sentence description for the entire document "
            "based on the tree structure and node summaries."
        )

    return json.dumps(result)
