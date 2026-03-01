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

from scout_ai.exceptions import ExtractionError, LLMClientError, RetrievalError
from scout_ai.hooks.run_tracker import get_current_run, track_stage
from scout_ai.models import (
    BatchExtractionResult,
    DocumentIndex,
    ExtractionQuestion,
    PageContent,
    RetrievalResult,
)

if TYPE_CHECKING:
    from scout_ai.core.config import AppSettings
    from scout_ai.domains.aps.validation.engine import RulesEngine
    from scout_ai.interfaces.chat import IChatProvider
    from scout_ai.interfaces.retrieval import IRetrievalProvider
    from scout_ai.validation.models import ValidationReport

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

        # Step 1: Category-batched retrieval (resilient)
        retrieval_results: dict[str, RetrievalResult] = {}
        try:
            with track_stage("retrieval") as stage:
                retrieval_results = await self._retrieval.batch_retrieve(index, questions)
                stage.success_count = len(retrieval_results)
        except (RetrievalError, LLMClientError) as e:
            log.error("Batch retrieval failed, degrading gracefully: %s", e)
            run = get_current_run()
            if run:
                run.errors.append(f"retrieval_failed: {e}")

        # Step 2: Extract answers per category (str-keyed)
        by_category: dict[str, list[ExtractionQuestion]] = defaultdict(list)
        for q in questions:
            by_category[q.category].append(q)

        results: list[BatchExtractionResult] = []
        for category_str, cat_questions in by_category.items():
            retrieval = retrieval_results.get(category_str)
            if not retrieval or not retrieval.retrieved_nodes:
                log.warning("No retrieval results for category %s", category_str)
                results.append(
                    BatchExtractionResult(
                        category=category_str,
                        retrieval=retrieval or RetrievalResult(query=category_str),
                    )
                )
                continue

            context = build_cited_context(retrieval.retrieved_nodes, page_map)
            if not context:
                results.append(BatchExtractionResult(category=category_str, retrieval=retrieval))
                continue

            try:
                with track_stage(f"extraction:{category_str}") as stage:
                    extractions = await self._chat.extract_answers(cat_questions, context)
                    stage.success_count = len(extractions)
            except (ExtractionError, LLMClientError) as e:
                log.error("Extraction failed for %s: %s", category_str, e)
                extractions = []
                run = get_current_run()
                if run:
                    run.errors.append(f"extraction_failed:{category_str}: {e}")

            results.append(
                BatchExtractionResult(
                    category=category_str,
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
        rules_engine: RulesEngine | None = None,
        synthesis_pipeline: Any | None = None,
    ) -> tuple[list[BatchExtractionResult], Any | None, ValidationReport | None]:
        """Full pipeline: retrieve -> extract -> synthesize -> validate.

        Args:
            index: Document index for retrieval.
            questions: Questions to answer.
            pages: Optional raw pages for per-page citation markers.
            synthesize: Whether to run the synthesis step.
            document_metadata: Optional metadata passed to synthesis.
            rules_engine: Optional validation engine for post-synthesis checks.
            synthesis_pipeline: Optional domain synthesis pipeline. If None,
                falls back to APS SynthesisPipeline for backward compatibility.

        Returns:
            Tuple of (extraction results, optional summary,
            optional validation report).
        """
        batch_results = await self.run(index, questions, pages=pages)

        summary: Any | None = None
        validation_report: ValidationReport | None = None
        if synthesize:
            if synthesis_pipeline is None:
                from scout_ai.domains.aps.synthesis.pipeline import SynthesisPipeline

                cache_enabled = getattr(self._chat, "_cache_enabled", False)
                client = getattr(self._chat, "_client", None)
                if client is None:
                    log.warning("Cannot synthesize: chat provider has no _client attribute")
                    return batch_results, None, None
                synthesis_pipeline = SynthesisPipeline(client, cache_enabled=cache_enabled)

            metadata = document_metadata or {"doc_id": index.doc_id, "doc_name": index.doc_name}
            try:
                with track_stage("synthesis"):
                    summary = await synthesis_pipeline.synthesize(batch_results, metadata)
            except Exception as e:
                log.error("Synthesis failed, returning None: %s", e)
                summary = None
                run = get_current_run()
                if run:
                    run.errors.append(f"synthesis_failed: {e}")

        return batch_results, summary, validation_report


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
        start = node["start_index"]
        end = node["end_index"]
        header = "[Section: %s | Type: %s | Pages %d-%d]" % (title, ctype, start, end)

        if page_map and start != end:
            page_parts: list[str] = []
            for pg in range(start, end + 1):
                pg_text = page_map.get(pg)
                if pg_text:
                    page_parts.append("[Page %d]\n%s" % (pg, pg_text))
            if page_parts:
                parts.append("%s\n%s" % (header, "\n\n".join(page_parts)))
            else:
                parts.append("%s\n[Page %d]\n%s" % (header, start, text))
        else:
            parts.append("%s\n[Page %d]\n%s" % (header, start, text))

    return "\n\n".join(parts)


def create_extraction_pipeline(settings: AppSettings) -> ExtractionPipeline:
    """Create an ExtractionPipeline using the legacy providers.

    This factory uses the existing provider classes until the full
    Strands agent migration is complete. When ``settings.domain`` is
    configured, domain-specific prompts and category descriptions are
    injected from the domain registry into the providers.
    """
    from scout_ai.config import ScoutSettings
    from scout_ai.inference import create_inference_backend
    from scout_ai.providers.pageindex.chat import ScoutChat
    from scout_ai.providers.pageindex.client import LLMClient
    from scout_ai.providers.pageindex.retrieval import ScoutRetrieval

    legacy_settings = ScoutSettings(
        llm_base_url=settings.llm.base_url,
        llm_api_key=settings.llm.api_key,
        llm_model=settings.llm.model,
        llm_temperature=settings.llm.temperature,
        llm_top_p=settings.llm.top_p,
        llm_seed=settings.llm.seed,
        llm_timeout=settings.llm.timeout,
        llm_max_retries=settings.llm.max_retries,
        retry_jitter_factor=settings.llm.retry_jitter_factor,
        retry_max_delay=settings.llm.retry_max_delay,
        retrieval_max_concurrent=settings.retrieval.max_concurrent,
        retrieval_top_k_nodes=settings.retrieval.top_k_nodes,
    )

    # Resolve domain-specific prompts and categories from registry
    category_descriptions: dict[str, str] | None = None
    try:
        from scout_ai.domains.registry import get_registry

        domain_config = get_registry().get(settings.domain)
        category_descriptions = domain_config.category_descriptions or None
    except (KeyError, ImportError):
        pass

    backend = create_inference_backend(settings)
    client = LLMClient(legacy_settings, backend=backend)
    retrieval = ScoutRetrieval(
        legacy_settings,
        client,
        category_descriptions=category_descriptions,
        domain=settings.domain,
    )
    chat = ScoutChat(legacy_settings, client, domain=settings.domain)

    return ExtractionPipeline(retrieval, chat)
