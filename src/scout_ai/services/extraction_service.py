"""Extraction service: orchestrates batch retrieval + answer extraction."""

from __future__ import annotations

import logging
from typing import Any

from scout_ai.interfaces.chat import IChatProvider
from scout_ai.interfaces.retrieval import IRetrievalProvider
from scout_ai.models import (
    BatchExtractionResult,
    DocumentIndex,
    ExtractionCategory,
    ExtractionQuestion,
    PageContent,
    RetrievalResult,
)

log = logging.getLogger(__name__)


class ExtractionService:
    """Retrieve relevant context by category, then extract answers."""

    def __init__(
        self,
        retrieval_provider: IRetrievalProvider,
        chat_provider: IChatProvider,
    ) -> None:
        self._retrieval = retrieval_provider
        self._chat = chat_provider

    async def extract(
        self,
        index: DocumentIndex,
        questions: list[ExtractionQuestion],
        *,
        pages: list[PageContent] | None = None,
    ) -> list[BatchExtractionResult]:
        """Run the full extraction pipeline: batch retrieve → extract answers.

        Args:
            pages: Optional raw page content for per-page citation markers.
                   When provided, multi-page nodes get [Page N] markers so
                   the LLM can cite specific pages.
        """
        # Build page lookup for per-page context markers
        page_map: dict[int, str] = {}
        if pages:
            page_map = {p.page_number: p.text for p in pages}

        # Step 1: Category-batched retrieval
        retrieval_results = await self._retrieval.batch_retrieve(index, questions)

        # Step 2: Extract answers per category
        results: list[BatchExtractionResult] = []
        from collections import defaultdict
        by_category: dict[ExtractionCategory, list[ExtractionQuestion]] = defaultdict(list)
        for q in questions:
            by_category[q.category].append(q)

        for category, cat_questions in by_category.items():
            retrieval = retrieval_results.get(category)
            if not retrieval or not retrieval.retrieved_nodes:
                log.warning(f"No retrieval results for category {category.value}")
                results.append(
                    BatchExtractionResult(
                        category=category,
                        retrieval=retrieval or RetrievalResult(query=category.value),
                    )
                )
                continue

            context = self._build_cited_context(retrieval.retrieved_nodes, page_map)

            if not context:
                results.append(BatchExtractionResult(category=category, retrieval=retrieval))
                continue

            extractions = await self._chat.extract_answers(cat_questions, context)
            results.append(
                BatchExtractionResult(
                    category=category,
                    retrieval=retrieval,
                    extractions=extractions,
                )
            )

        return results

    @staticmethod
    def _build_cited_context(
        nodes: list[dict[str, Any]],
        page_map: dict[int, str],
    ) -> str:
        """Build extraction context with per-page markers for citation grounding.

        When *page_map* is populated, multi-page nodes are split into per-page
        blocks with ``[Page N]`` markers so the LLM can cite exact pages.
        """
        parts: list[str] = []
        for node in nodes:
            text = node.get("text", "")
            if not text:
                continue

            title = node.get("title", "")
            ctype = node.get("content_type", "unknown")
            # Normalise enum values to string
            if hasattr(ctype, "value"):
                ctype = ctype.value
            start = node["start_index"]
            end = node["end_index"]
            header = f"[Section: {title} | Type: {ctype} | Pages {start}-{end}]"

            if page_map and start != end:
                # Multi-page node: emit per-page text from raw pages
                page_parts: list[str] = []
                for pg in range(start, end + 1):
                    pg_text = page_map.get(pg)
                    if pg_text:
                        page_parts.append(f"[Page {pg}]\n{pg_text}")
                if page_parts:
                    parts.append(f"{header}\n" + "\n\n".join(page_parts))
                else:
                    # Fallback: pages not available in map
                    parts.append(f"{header}\n[Page {start}]\n{text}")
            else:
                # Single-page node or no page_map
                parts.append(f"{header}\n[Page {start}]\n{text}")

        return "\n\n".join(parts)
