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

from pageindex_rag.aps.categories import CATEGORY_DESCRIPTIONS
from pageindex_rag.aps.prompts import CATEGORY_SEARCH_PROMPT
from pageindex_rag.config import PageIndexSettings
from pageindex_rag.models import (
    DocumentIndex,
    ExtractionCategory,
    ExtractionQuestion,
    RetrievalResult,
)
from pageindex_rag.providers.pageindex.client import LLMClient
from pageindex_rag.providers.pageindex.retrieval import PageIndexRetrieval
from pageindex_rag.providers.pageindex.tree_utils import (
    create_node_mapping,
    get_source_pages,
    tree_to_dict,
)

log = logging.getLogger(__name__)


class BatchRetrieval:
    """Batches questions by category for efficient tree search."""

    def __init__(
        self,
        settings: PageIndexSettings,
        client: LLMClient,
        retrieval: PageIndexRetrieval,
    ) -> None:
        self._settings = settings
        self._client = client
        self._retrieval = retrieval

    async def batch_retrieve(
        self,
        index: DocumentIndex,
        questions: list[ExtractionQuestion],
    ) -> dict[ExtractionCategory, RetrievalResult]:
        """Group questions by category and run one search per category.

        Returns a dict mapping each category to its retrieval result.
        """
        # Group questions by category
        by_category: dict[ExtractionCategory, list[ExtractionQuestion]] = defaultdict(list)
        for q in questions:
            by_category[q.category].append(q)

        log.info(
            f"Batch retrieval: {len(questions)} questions in {len(by_category)} categories"
        )

        # Run one search per category with concurrency control
        sem = asyncio.Semaphore(self._settings.retrieval_max_concurrent)
        results: dict[ExtractionCategory, RetrievalResult] = {}

        async def _search_category(
            category: ExtractionCategory,
            cat_questions: list[ExtractionQuestion],
        ) -> tuple[ExtractionCategory, RetrievalResult]:
            async with sem:
                result = await self._category_search(index, category, cat_questions)
                return category, result

        tasks = [
            _search_category(cat, qs) for cat, qs in by_category.items()
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in raw_results:
            if isinstance(result, Exception):
                log.warning(f"Category search failed: {result}")
                continue
            category, retrieval_result = result
            results[category] = retrieval_result

        return results

    async def _category_search(
        self,
        index: DocumentIndex,
        category: ExtractionCategory,
        questions: list[ExtractionQuestion],
    ) -> RetrievalResult:
        """Build a synthesized query for a category and search the tree."""
        category_desc = CATEGORY_DESCRIPTIONS.get(category, category.value)

        # Build a synthesized query from category description
        tree_structure = json.dumps(tree_to_dict(index.tree), indent=2)
        prompt = CATEGORY_SEARCH_PROMPT.format(
            category=category.value,
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

        synthesized_query = f"[{category.value}] {category_desc}"
        return RetrievalResult(
            query=synthesized_query,
            retrieved_nodes=retrieved_nodes,
            source_pages=source_pages,
            reasoning=reasoning,
        )
