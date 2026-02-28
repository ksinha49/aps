"""Batch retrieval skill — category-grouped multi-question retrieval.

Replaces ``providers/pageindex/batch_retrieval.py``. Groups questions by
ExtractionCategory and runs one synthesized search per category.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from pageindex_rag.aps.categories import CATEGORY_DESCRIPTIONS
from pageindex_rag.models import (
    DocumentIndex,
    ExtractionCategory,
    ExtractionQuestion,
    RetrievalResult,
)
from pageindex_rag.providers.pageindex.tree_utils import (
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

    # Group by category
    by_category: dict[ExtractionCategory, list[ExtractionQuestion]] = defaultdict(list)
    for q in questions:
        by_category[q.category].append(q)

    tree_structure = json.dumps(tree_to_dict(index.tree), indent=2)

    settings = tool_context.invocation_state.get("settings")  # type: ignore[union-attr]
    top_k = 5
    if settings:
        top_k = settings.retrieval.top_k_nodes

    category_tasks = []
    for category, cat_questions in by_category.items():
        category_desc = CATEGORY_DESCRIPTIONS.get(category, category.value)
        category_tasks.append({
            "category": category.value,
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
) -> dict[ExtractionCategory, RetrievalResult]:
    """Pure logic: resolve category-level results to RetrievalResults.

    Args:
        index: The document index.
        category_results: Dict mapping category value string to
            {'node_ids': [...], 'reasoning': '...'}.
        top_k: Maximum nodes per category.

    Returns:
        Dict mapping ExtractionCategory to RetrievalResult.
    """
    node_map = create_node_mapping(index.tree)
    results: dict[ExtractionCategory, RetrievalResult] = {}

    for cat_str, result_data in category_results.items():
        try:
            category = ExtractionCategory(cat_str)
        except ValueError:
            log.warning(f"Unknown category: {cat_str}")
            continue

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
        category_desc = CATEGORY_DESCRIPTIONS.get(category, category.value)

        results[category] = RetrievalResult(
            query=f"[{category.value}] {category_desc}",
            retrieved_nodes=retrieved_nodes,
            source_pages=source_pages,
            reasoning=reasoning,
        )

    return results


def _ensure_questions(questions_raw: list[Any]) -> list[ExtractionQuestion]:
    """Convert raw question data to ExtractionQuestion list."""
    questions: list[ExtractionQuestion] = []
    for q in questions_raw:
        if isinstance(q, ExtractionQuestion):
            questions.append(q)
        elif isinstance(q, dict):
            questions.append(ExtractionQuestion(**q))
    return questions
