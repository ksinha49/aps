"""Orchestrator: wires agents into multi-step pipelines.

Replaces ``ExtractionService`` manual orchestration with a sequential
pipeline pattern: retrieve then extract.

Uses simple function composition rather than Strands Graph (which requires
agents that communicate via natural language). This approach preserves
the structured data passing between retrieval and extraction.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from scout_ai.models import (
    BatchExtractionResult,
    DocumentIndex,
    ExtractionCategory,
    ExtractionQuestion,
    PageContent,
    RetrievalResult,
)
from scout_ai.synthesis.models import UnderwriterSummary

if TYPE_CHECKING:
    from scout_ai.core.config import AppSettings
    from scout_ai.interfaces.chat import IChatProvider
    from scout_ai.interfaces.retrieval import IRetrievalProvider

log = logging.getLogger(__name__)


class ExtractionPipeline:
    """Sequential pipeline: batch_retrieve then extract_answers.

    This orchestrator preserves the existing data flow while being
    compatible with both legacy providers and Strands agents.
    """

    def __init__(
        self,
        retrieval_provider: IRetrievalProvider,
        chat_provider: IChatProvider,
    ) -> None:
        self._retrieval = retrieval_provider
        self._chat = chat_provider

    async def run(
        self,
        index: DocumentIndex,
        questions: list[ExtractionQuestion],
        *,
        pages: list[PageContent] | None = None,
    ) -> list[BatchExtractionResult]:
        """Run the full extraction pipeline.

        Args:
            index: Document index for retrieval.
            questions: Questions to answer.
            pages: Optional raw pages for per-page citation markers.

        Returns:
            Per-category extraction results.
        """
        page_map: dict[int, str] = {}
        if pages:
            page_map = {p.page_number: p.text for p in pages}

        # Step 1: Category-batched retrieval
        retrieval_results = await self._retrieval.batch_retrieve(index, questions)

        # Step 2: Extract answers per category
        by_category: dict[ExtractionCategory, list[ExtractionQuestion]] = defaultdict(list)
        for q in questions:
            by_category[q.category].append(q)

        results: list[BatchExtractionResult] = []
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

            context = build_cited_context(retrieval.retrieved_nodes, page_map)
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

    async def run_with_synthesis(
        self,
        index: DocumentIndex,
        questions: list[ExtractionQuestion],
        *,
        pages: list[PageContent] | None = None,
        synthesize: bool = True,
        document_metadata: dict[str, Any] | None = None,
    ) -> tuple[list[BatchExtractionResult], UnderwriterSummary | None]:
        """Full pipeline: retrieve -> extract -> synthesize.

        Args:
            index: Document index for retrieval.
            questions: Questions to answer.
            pages: Optional raw pages for per-page citation markers.
            synthesize: Whether to run the synthesis step.
            document_metadata: Optional metadata passed to synthesis.

        Returns:
            Tuple of (extraction results, optional underwriter summary).
        """
        batch_results = await self.run(index, questions, pages=pages)

        summary: UnderwriterSummary | None = None
        if synthesize:
            from scout_ai.synthesis.pipeline import SynthesisPipeline

            # Determine cache_enabled from chat provider if available
            cache_enabled = getattr(self._chat, "_cache_enabled", False)
            client = getattr(self._chat, "_client", None)
            if client is None:
                log.warning("Cannot synthesize: chat provider has no _client attribute")
                return batch_results, None

            synth = SynthesisPipeline(client, cache_enabled=cache_enabled)
            metadata = document_metadata or {"doc_id": index.doc_id, "doc_name": index.doc_name}
            summary = await synth.synthesize(batch_results, metadata)

        return batch_results, summary


def build_cited_context(
    nodes: list[dict[str, Any]],
    page_map: dict[int, str],
) -> str:
    """Build extraction context with per-page markers for citation grounding.

    When page_map is populated, multi-page nodes are split into per-page
    blocks with [Page N] markers so the LLM can cite exact pages.
    """
    parts: list[str] = []
    for node in nodes:
        text = node.get("text", "")
        if not text:
            continue

        title = node.get("title", "")
        ctype = node.get("content_type", "unknown")
        if hasattr(ctype, "value"):
            ctype = ctype.value
        start = node["start_index"]
        end = node["end_index"]
        header = f"[Section: {title} | Type: {ctype} | Pages {start}-{end}]"

        if page_map and start != end:
            page_parts: list[str] = []
            for pg in range(start, end + 1):
                pg_text = page_map.get(pg)
                if pg_text:
                    page_parts.append(f"[Page {pg}]\n{pg_text}")
            if page_parts:
                parts.append(f"{header}\n" + "\n\n".join(page_parts))
            else:
                parts.append(f"{header}\n[Page {start}]\n{text}")
        else:
            parts.append(f"{header}\n[Page {start}]\n{text}")

    return "\n\n".join(parts)


def create_extraction_pipeline(settings: AppSettings) -> ExtractionPipeline:
    """Create an ExtractionPipeline using the legacy providers.

    This factory uses the existing provider classes until the full
    Strands agent migration is complete.
    """
    from scout_ai.config import ScoutSettings
    from scout_ai.providers.pageindex.chat import ScoutChat
    from scout_ai.providers.pageindex.client import LLMClient
    from scout_ai.providers.pageindex.retrieval import ScoutRetrieval

    # Bridge new AppSettings to legacy ScoutSettings
    legacy_settings = ScoutSettings(
        llm_base_url=settings.llm.base_url,
        llm_api_key=settings.llm.api_key,
        llm_model=settings.llm.model,
        llm_temperature=settings.llm.temperature,
        llm_timeout=settings.llm.timeout,
        llm_max_retries=settings.llm.max_retries,
        retrieval_max_concurrent=settings.retrieval.max_concurrent,
        retrieval_top_k_nodes=settings.retrieval.top_k_nodes,
    )

    client = LLMClient(legacy_settings)
    retrieval = ScoutRetrieval(legacy_settings, client)
    chat = ScoutChat(legacy_settings, client)

    return ExtractionPipeline(retrieval, chat)
