"""Synthesis pipeline: aggregates 900 extraction results into a structured underwriter summary."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from scout_ai.models import BatchExtractionResult
from scout_ai.prompts.registry import get_prompt
from scout_ai.providers.pageindex.client import LLMClient
from scout_ai.synthesis.models import SynthesisSection, UnderwriterSummary

log = logging.getLogger(__name__)


class SynthesisPipeline:
    """Aggregates extraction results into a structured underwriter summary.

    The pipeline works in two phases:
    1. **Prepare**: Filter to high-confidence answers, group by category
    2. **Generate**: LLM synthesizes category summaries into a narrative report

    When ``cache_enabled`` is True, the synthesis system prompt is cached
    for reuse across multiple documents processed in the same session.
    """

    def __init__(self, client: LLMClient, *, cache_enabled: bool = False) -> None:
        self._client = client
        self._cache_enabled = cache_enabled

    async def synthesize(
        self,
        extraction_results: list[BatchExtractionResult],
        document_metadata: dict[str, Any] | None = None,
    ) -> UnderwriterSummary:
        """Produce an underwriter summary from extraction results.

        Args:
            extraction_results: Per-category extraction results from the pipeline.
            document_metadata: Optional metadata (doc_id, doc_name, etc.).

        Returns:
            Structured ``UnderwriterSummary`` with sections, risk factors,
            and overall assessment.
        """
        metadata = document_metadata or {}

        # Phase 1: Prepare per-category summaries
        category_summaries = self._prepare_category_summaries(extraction_results)

        # Phase 2: Generate narrative via LLM
        system_prompt_text = get_prompt("aps", "synthesis", "SYNTHESIS_SYSTEM_PROMPT")
        synthesis_prompt_template = get_prompt("aps", "synthesis", "SYNTHESIS_PROMPT")

        system_prompt: str | None = system_prompt_text if self._cache_enabled else None

        prompt_body = synthesis_prompt_template.format(
            category_summaries=json.dumps(category_summaries, indent=2),
            document_metadata=json.dumps(metadata),
        )

        if not self._cache_enabled:
            # Inline system prompt into user message when caching is off
            prompt_body = f"{system_prompt_text}\n\n{prompt_body}"

        response = await self._client.complete(
            prompt_body,
            system_prompt=system_prompt,
            cache_system=self._cache_enabled,
        )

        return self._parse_summary(response, extraction_results, metadata)

    def _prepare_category_summaries(
        self,
        results: list[BatchExtractionResult],
    ) -> list[dict[str, Any]]:
        """Filter to high-confidence answers, group by category."""
        summaries: list[dict[str, Any]] = []
        for batch in results:
            high_conf = [
                e for e in (batch.extractions or [])
                if e.confidence >= 0.7
            ]
            summaries.append({
                "category": batch.category.value,
                "question_count": len(batch.extractions or []),
                "high_confidence_count": len(high_conf),
                "answers": [
                    {
                        "question_id": e.question_id,
                        "answer": e.answer,
                        "confidence": e.confidence,
                        "citations": [c.verbatim_quote for c in e.citations[:2]],
                    }
                    for e in high_conf
                ],
            })
        return summaries

    def _parse_summary(
        self,
        response: str,
        results: list[BatchExtractionResult],
        metadata: dict[str, Any],
    ) -> UnderwriterSummary:
        """Parse LLM response into an UnderwriterSummary."""
        parsed = self._client.extract_json(response)

        total_answered = sum(
            len(batch.extractions or []) for batch in results
        )
        high_conf_count = sum(
            sum(1 for e in (batch.extractions or []) if e.confidence >= 0.7)
            for batch in results
        )

        sections = [
            SynthesisSection(
                title=s.get("title", ""),
                content=s.get("content", ""),
                source_categories=s.get("source_categories", []),
                key_findings=s.get("key_findings", []),
            )
            for s in parsed.get("sections", [])
        ]

        return UnderwriterSummary(
            document_id=metadata.get("doc_id", ""),
            patient_demographics=parsed.get("patient_demographics", ""),
            sections=sections,
            risk_factors=parsed.get("risk_factors", []),
            overall_assessment=parsed.get("overall_assessment", ""),
            total_questions_answered=total_answered,
            high_confidence_count=high_conf_count,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
