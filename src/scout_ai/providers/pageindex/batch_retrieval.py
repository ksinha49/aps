"""Category-batched multi-question retrieval.

Groups 900 questions into 16 extraction categories and runs one synthesized
tree search per category, reducing LLM calls from 900+ to 16.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import Any

from scout_ai.config import ScoutSettings
from scout_ai.models import (
    DocumentIndex,
    ExtractionQuestion,
    RetrievalResult,
)
from scout_ai.providers.pageindex.client import LLMClient
from scout_ai.providers.pageindex.retrieval import ScoutRetrieval
from scout_ai.providers.pageindex.tree_utils import (
    create_node_mapping,
    get_source_pages,
    tree_to_dict,
)

log = logging.getLogger(__name__)


class BatchRetrieval:
    """Batches questions by category for efficient tree search."""

    def __init__(
        self,
        settings: ScoutSettings,
        client: LLMClient,
        retrieval: ScoutRetrieval,
        *,
        category_descriptions: dict[str, str] | None = None,
        category_search_prompt: str | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._retrieval = retrieval
        self._category_descriptions = category_descriptions or {}
        self._category_search_prompt = category_search_prompt or ""

    async def batch_retrieve(
        self,
        index: DocumentIndex,
        questions: list[ExtractionQuestion],
    ) -> dict[str, RetrievalResult]:
        """Group questions by category and run one search per category.

        Returns a dict mapping each category string to its retrieval result.
        """
        # Group questions by category (str-keyed)
        by_category: dict[str, list[ExtractionQuestion]] = defaultdict(list)
        for q in questions:
            by_category[q.category].append(q)

        log.info(
            f"Batch retrieval: {len(questions)} questions in {len(by_category)} categories"
        )

        # Run one search per category with concurrency control
        sem = asyncio.Semaphore(self._settings.retrieval_max_concurrent)
        results: dict[str, RetrievalResult] = {}

        async def _search_category(
            category_str: str,
            cat_questions: list[ExtractionQuestion],
        ) -> tuple[str, RetrievalResult]:
            async with sem:
                result = await self._category_search(index, category_str, cat_questions)
                return category_str, result

        tasks = [
            _search_category(cat, qs) for cat, qs in by_category.items()
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in raw_results:
            if isinstance(result, Exception):
                log.warning(f"Category search failed: {result}")
                continue
            category_str, retrieval_result = result
            results[category_str] = retrieval_result

        return results

    async def _category_search(
        self,
        index: DocumentIndex,
        category_str: str,
        questions: list[ExtractionQuestion],
    ) -> RetrievalResult:
        """Build a synthesized query for a category and search the tree."""
        category_desc = self._category_descriptions.get(category_str, category_str)

        # Build a synthesized query from category description
        tree_structure = json.dumps(tree_to_dict(index.tree), indent=2)
        if not self._category_search_prompt:
            from scout_ai.prompts.registry import get_prompt

            self._category_search_prompt = get_prompt("aps", "retrieval", "CATEGORY_SEARCH_PROMPT")
        prompt = self._category_search_prompt.format(
            category=category_str,
            category_description=category_desc,
            tree_structure=tree_structure,
            top_k=self._settings.retrieval_top_k_nodes,
        )

        response = await self._client.complete(prompt)
        parsed = self._client.extract_json(response)

        node_ids = parsed.get("node_ids", [])
        reasoning = parsed.get("reasoning", "")

        # Resolve nodes
        node_map = create_node_mapping(index.tree)
        retrieved_nodes: list[dict[str, Any]] = []
        matched_nodes = []

        for nid in node_ids[: self._settings.retrieval_top_k_nodes]:
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

        synthesized_query = f"[{category_str}] {category_desc}"
        return RetrievalResult(
            query=synthesized_query,
            retrieved_nodes=retrieved_nodes,
            source_pages=source_pages,
            reasoning=reasoning,
        )
