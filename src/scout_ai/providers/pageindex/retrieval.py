"""Single-query tree search retrieval.

Uses LLM reasoning over the tree structure (titles, summaries, content_types)
to identify the most relevant nodes for a query.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from scout_ai.config import ScoutSettings
from scout_ai.interfaces.retrieval import IRetrievalProvider
from scout_ai.models import (
    DocumentIndex,
    ExtractionQuestion,
    RetrievalResult,
)
from scout_ai.providers.pageindex.client import LLMClient
from scout_ai.providers.pageindex.tree_utils import (
    create_node_mapping,
    get_source_pages,
    tree_to_dict,
)

log = logging.getLogger(__name__)


class ScoutRetrieval(IRetrievalProvider):
    """Tree-based retrieval: LLM reasons over document structure to find relevant nodes."""

    def __init__(
        self,
        settings: ScoutSettings,
        client: LLMClient,
        *,
        tree_search_prompt: str | None = None,
        category_descriptions: dict[str, str] | None = None,
        category_search_prompt: str | None = None,
        domain: str = "aps",
    ) -> None:
        self._settings = settings
        self._client = client
        self._tree_search_prompt = tree_search_prompt or ""
        self._category_descriptions = category_descriptions
        self._category_search_prompt = category_search_prompt
        self._domain = domain

    async def retrieve(
        self,
        index: DocumentIndex,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Search the tree for nodes relevant to *query*."""
        tree_structure = json.dumps(tree_to_dict(index.tree), indent=2)
        if not self._tree_search_prompt:
            from scout_ai.prompts.registry import get_prompt

            self._tree_search_prompt = get_prompt(self._domain, "retrieval", "TREE_SEARCH_PROMPT")
        prompt = self._tree_search_prompt.format(
            tree_structure=tree_structure,
            query=query,
            top_k=top_k,
        )

        response = await self._client.complete(prompt)
        parsed = self._client.extract_json(response)

        # Handle malformed LLM output: model may return a list instead of dict
        if isinstance(parsed, list):
            node_ids = parsed
            reasoning = ""
        else:
            node_ids = parsed.get("node_ids", [])
            reasoning = parsed.get("reasoning", "")

        # Coerce node_ids: LLM may return dicts like {"node_id": "..."} instead of strings
        clean_ids: list[str] = []
        for nid in node_ids:
            if isinstance(nid, str):
                clean_ids.append(nid)
            elif isinstance(nid, dict) and "node_id" in nid:
                clean_ids.append(str(nid["node_id"]))

        # Resolve node_ids to actual nodes
        node_map = create_node_mapping(index.tree)
        retrieved_nodes: list[dict[str, Any]] = []
        matched_nodes = []

        for nid in clean_ids[:top_k]:
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

    async def batch_retrieve(
        self,
        index: DocumentIndex,
        questions: list[ExtractionQuestion],
    ) -> dict[str, RetrievalResult]:
        """Category-batched retrieval — delegates to BatchRetrieval."""
        from scout_ai.providers.pageindex.batch_retrieval import BatchRetrieval

        batch = BatchRetrieval(
            self._settings,
            self._client,
            self,
            category_descriptions=self._category_descriptions,
            category_search_prompt=self._category_search_prompt,
            domain=self._domain,
        )
        results = await batch.batch_retrieve(index, questions)
        return results
