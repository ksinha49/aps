"""Batch retrieval skill — category-grouped multi-question retrieval.

Replaces ``providers/pageindex/batch_retrieval.py``. Groups questions by
category and runs one synthesized search per category.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from scout_ai.models import (
    DocumentIndex,
    ExtractionQuestion,
    RetrievalResult,
)
from scout_ai.providers.pageindex.tree_utils import (
    create_node_mapping,
    get_source_pages,
    tree_to_dict,
)

log = logging.getLogger(__name__)


@tool(context=True)
def batch_retrieve(  # type: ignore[assignment]
    tool_context: ToolContext = None,
) -> str:
    """Group extraction questions by category and prepare batch tree searches.

    Reads questions from invocation_state['questions'] and the document
    index from invocation_state['document_index'].

    Returns:
        JSON with per-category search tasks for the agent to execute.
    """
    index: DocumentIndex | None = tool_context.invocation_state.get("document_index")  # type: ignore[union-attr]
    questions_raw = tool_context.invocation_state.get("questions", [])  # type: ignore[union-attr]

    if not index:
        return json.dumps({"error": "No document_index in invocation state"})

    questions = _ensure_questions(questions_raw)
    if not questions:
        return json.dumps({"error": "No questions in invocation state"})

    # Group by category (str-based)
    by_category: dict[str, list[ExtractionQuestion]] = defaultdict(list)
    for q in questions:
        cat_key = q.category
        if hasattr(cat_key, "value"):
            cat_key = cat_key.value
        by_category[cat_key].append(q)

    tree_structure = json.dumps(tree_to_dict(index.tree), indent=2)

    settings = tool_context.invocation_state.get("settings")  # type: ignore[union-attr]
    top_k = 5
    if settings:
        top_k = settings.retrieval.top_k_nodes

    # Resolve category descriptions from domain registry
    category_descs = _get_category_descriptions(tool_context)

    category_tasks = []
    for cat_str, cat_questions in by_category.items():
        category_desc = category_descs.get(cat_str, cat_str)
        category_tasks.append({
            "category": cat_str,
            "category_description": category_desc,
            "question_count": len(cat_questions),
            "question_ids": [q.question_id for q in cat_questions],
            "top_k": top_k,
        })

    return json.dumps({
        "action": "batch_retrieve",
        "total_questions": len(questions),
        "total_categories": len(by_category),
        "tree_structure": tree_structure,
        "category_tasks": category_tasks,
        "instruction": (
            "For each category, search the document tree to find sections "
            "most likely to contain that category of information. "
            "Return node_ids for each category."
        ),
    })


def resolve_batch_results(
    index: DocumentIndex,
    category_results: dict[str, dict[str, Any]],
    top_k: int = 5,
    category_descriptions: dict[str, str] | None = None,
) -> dict[str, RetrievalResult]:
    """Pure logic: resolve category-level results to RetrievalResults.

    Args:
        index: The document index.
        category_results: Dict mapping category value string to
            {'node_ids': [...], 'reasoning': '...'}.
        top_k: Maximum nodes per category.
        category_descriptions: Optional mapping of category str to
            human-readable descriptions.

    Returns:
        Dict mapping category string to RetrievalResult.
    """
    descs = category_descriptions or {}
    node_map = create_node_mapping(index.tree)
    results: dict[str, RetrievalResult] = {}

    for cat_str, result_data in category_results.items():
        node_ids = result_data.get("node_ids", [])
        reasoning = result_data.get("reasoning", "")

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

        category_desc = descs.get(cat_str, cat_str)

        results[cat_str] = RetrievalResult(
            query=f"[{cat_str}] {category_desc}",
            retrieved_nodes=retrieved_nodes,
            source_pages=source_pages,
            reasoning=reasoning,
        )

    return results


def _get_category_descriptions(tool_context: ToolContext) -> dict[str, str]:
    """Resolve category descriptions from invocation state or domain registry."""
    # Check invocation state first (set by pipeline)
    descs = tool_context.invocation_state.get("category_descriptions")  # type: ignore[union-attr]
    if descs:
        return descs

    # Fall back to domain registry
    settings = tool_context.invocation_state.get("settings")  # type: ignore[union-attr]
    domain_name = "aps"
    if settings and hasattr(settings, "domain"):
        domain_name = settings.domain

    try:
        from scout_ai.domains.registry import get_registry

        registry = get_registry()
        domain_config = registry.get(domain_name)
        return domain_config.category_descriptions
    except (KeyError, ImportError):
        return {}


def _ensure_questions(questions_raw: list[Any]) -> list[ExtractionQuestion]:
    """Convert raw question data to ExtractionQuestion list."""
    questions: list[ExtractionQuestion] = []
    for q in questions_raw:
        if isinstance(q, ExtractionQuestion):
            questions.append(q)
        elif isinstance(q, dict):
            questions.append(ExtractionQuestion(**q))
    return questions
