"""Single-query tree search retrieval.

Uses LLM reasoning over the tree structure (titles, summaries, content_types)
to identify the most relevant nodes for a query.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pageindex_rag.aps.prompts import TREE_SEARCH_PROMPT
from pageindex_rag.config import PageIndexSettings
from pageindex_rag.interfaces.retrieval import IRetrievalProvider
from pageindex_rag.models import (
    DocumentIndex,
    ExtractionCategory,
    ExtractionQuestion,
    RetrievalResult,
)
from pageindex_rag.providers.pageindex.client import LLMClient
from pageindex_rag.providers.pageindex.tree_utils import (
    create_node_mapping,
    get_source_pages,
    tree_to_dict,
)

log = logging.getLogger(__name__)


class PageIndexRetrieval(IRetrievalProvider):
    """Tree-based retrieval: LLM reasons over document structure to find relevant nodes."""

    def __init__(self, settings: PageIndexSettings, client: LLMClient) -> None:
        self._settings = settings
        self._client = client

    async def retrieve(
        self,
        index: DocumentIndex,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Search the tree for nodes relevant to *query*."""
        tree_structure = json.dumps(tree_to_dict(index.tree), indent=2)
        prompt = TREE_SEARCH_PROMPT.format(
            tree_structure=tree_structure,
            query=query,
            top_k=top_k,
        )

        response = await self._client.complete(prompt)
        parsed = self._client.extract_json(response)

        node_ids = parsed.get("node_ids", [])
        reasoning = parsed.get("reasoning", "")

        # Resolve node_ids to actual nodes
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

    async def batch_retrieve(
        self,
        index: DocumentIndex,
        questions: list[ExtractionQuestion],
    ) -> dict[ExtractionCategory, RetrievalResult]:
        """Category-batched retrieval — delegates to BatchRetrieval."""
        from pageindex_rag.providers.pageindex.batch_retrieval import BatchRetrieval

        batch = BatchRetrieval(self._settings, self._client, self)
        return await batch.batch_retrieve(index, questions)
