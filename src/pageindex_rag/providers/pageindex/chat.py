"""Extraction completions with tiered prompting and optional Anthropic prompt caching.

Tier 1 (simple lookup, ~60-70% of questions): batch 20 questions per prompt.
Tier 2/3 (cross-reference / reasoning): individual prompts with reasoning chains.

When caching is enabled, document context is placed in a system message with
``cache_control`` so it's cached server-side across all extraction calls.
"""

from __future__ import annotations

import logging
from typing import Any

from pageindex_rag.aps.prompts import BATCH_EXTRACTION_PROMPT, INDIVIDUAL_EXTRACTION_PROMPT
from pageindex_rag.config import PageIndexSettings
from pageindex_rag.interfaces.chat import IChatProvider
from pageindex_rag.models import Citation, ExtractionQuestion, ExtractionResult
from pageindex_rag.providers.pageindex.client import LLMClient

log = logging.getLogger(__name__)

TIER1_BATCH_SIZE = 20

_CACHED_SYSTEM_TEMPLATE = (
    "You are extracting information from an APS (Attending Physician Statement) medical record.\n"
    "Answer each question based ONLY on the provided context. If the answer is not found, say \"Not found\".\n\n"
    "Document Context:\n{context}"
)

_CACHED_BATCH_PROMPT = """Questions:
{questions}

For each question, return:
{{
    "answers": [
        {{
            "question_id": "<id>",
            "answer": "<extracted answer or 'Not found'>",
            "confidence": <0.0 to 1.0>,
            "citations": [
                {{
                    "page_number": <exact page number from [Page N] markers>,
                    "section_title": "<section name from section headers>",
                    "section_type": "<section type from section headers>",
                    "verbatim_quote": "<exact quote copied from the source text>"
                }}
            ]
        }},
        ...
    ]
}}

IMPORTANT: Each citation must include:
- The exact page number where the evidence appears (use the [Page N] markers in the context)
- The section title and type (from the [Section: ... | Type: ...] headers)
- A verbatim quote copied exactly from the source text — do not paraphrase

Directly return the final JSON structure. Do not output anything else."""

_CACHED_INDIVIDUAL_PROMPT = """Question: {question}

Think step-by-step:
1. Identify which parts of the context are relevant
2. Cross-reference dates, providers, and findings
3. Synthesize the answer

Return JSON:
{{
    "reasoning": "<step by step analysis>",
    "answer": "<extracted answer or 'Not found'>",
    "confidence": <0.0 to 1.0>,
    "citations": [
        {{
            "page_number": <exact page number from [Page N] markers>,
            "section_title": "<section name from section headers>",
            "section_type": "<section type from section headers>",
            "verbatim_quote": "<exact quote copied from the source text>"
        }}
    ]
}}

IMPORTANT: Each citation must include:
- The exact page number where the evidence appears (use the [Page N] markers in the context)
- The section title and type (from the [Section: ... | Type: ...] headers)
- A verbatim quote copied exactly from the source text — do not paraphrase

Directly return the final JSON structure. Do not output anything else."""


class PageIndexChat(IChatProvider):
    """Extraction chat provider with tiered prompting strategy."""

    def __init__(
        self,
        settings: PageIndexSettings,
        client: LLMClient,
        *,
        cache_enabled: bool = False,
    ) -> None:
        self._settings = settings
        self._client = client
        self._cache_enabled = cache_enabled

    async def extract_answers(
        self,
        questions: list[ExtractionQuestion],
        context: str,
    ) -> list[ExtractionResult]:
        """Extract answers using tiered prompting.

        - Tier 1: batch 20 questions per prompt
        - Tier 2/3: individual prompts with reasoning chains

        When ``cache_enabled`` is True, document context is placed in a cached
        system prompt instead of being repeated in every user message.
        """
        tier1 = [q for q in questions if q.tier == 1]
        tier_high = [q for q in questions if q.tier >= 2]

        # Build cacheable system prompt with document context
        system_prompt: str | None = None
        if self._cache_enabled:
            system_prompt = _CACHED_SYSTEM_TEMPLATE.format(context=context[:8000])

        results: list[ExtractionResult] = []

        # Tier 1: batch extraction
        for i in range(0, len(tier1), TIER1_BATCH_SIZE):
            batch = tier1[i : i + TIER1_BATCH_SIZE]
            batch_results = await self._extract_batch(
                batch,
                context,
                system_prompt=system_prompt,
                cache_system=self._cache_enabled,
            )
            results.extend(batch_results)

        # Tier 2/3: individual extraction
        for q in tier_high:
            result = await self._extract_individual(
                q,
                context,
                system_prompt=system_prompt,
                cache_system=self._cache_enabled,
            )
            results.append(result)

        return results

    async def chat(self, query: str, context: str) -> str:
        """Free-form completion over context."""
        prompt = f"""Based on the following medical record context, answer the question.

Context:
{context}

Question: {query}

Provide a detailed answer based only on the provided context."""
        return await self._client.complete(prompt)

    # ── Citation helpers ────────────────────────────────────────────

    @staticmethod
    def _parse_citations(
        raw_citations: list[dict[str, Any]],
    ) -> tuple[list[Citation], list[int], str]:
        """Parse citation dicts into models and derive backward-compat fields."""
        citations: list[Citation] = []
        pages: set[int] = set()
        quotes: list[str] = []
        for c in raw_citations:
            cit = Citation(
                page_number=int(c.get("page_number", 0)),
                section_title=c.get("section_title", ""),
                section_type=c.get("section_type", ""),
                verbatim_quote=c.get("verbatim_quote", ""),
            )
            citations.append(cit)
            if cit.page_number:
                pages.add(cit.page_number)
            if cit.verbatim_quote:
                quotes.append(cit.verbatim_quote)
        return citations, sorted(pages), "; ".join(quotes)

    @staticmethod
    def _citations_from_answer(ans: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract raw citation list from an LLM answer, with old-format fallback."""
        raw = ans.get("citations", [])
        if raw:
            return raw
        # Fallback: convert old source_pages/evidence_text format
        old_pages = ans.get("source_pages", [])
        if old_pages:
            return [
                {
                    "page_number": p,
                    "verbatim_quote": ans.get("evidence_text", ""),
                }
                for p in old_pages
            ]
        return []

    # ── Tier 1: Batch extraction ─────────────────────────────────────

    async def _extract_batch(
        self,
        questions: list[ExtractionQuestion],
        context: str,
        *,
        system_prompt: str | None = None,
        cache_system: bool = False,
    ) -> list[ExtractionResult]:
        """Extract answers for a batch of Tier 1 questions."""
        questions_text = "\n".join(
            f"- [{q.question_id}] {q.question_text}" for q in questions
        )

        if system_prompt:
            # Caching path: context is in system prompt, user message has only questions
            prompt = _CACHED_BATCH_PROMPT.format(questions=questions_text)
        else:
            # Legacy path: context + questions in a single user message
            prompt = BATCH_EXTRACTION_PROMPT.format(
                context=context[:8000],
                questions=questions_text,
            )

        response = await self._client.complete(
            prompt,
            system_prompt=system_prompt,
            cache_system=cache_system,
        )
        parsed = self._client.extract_json(response)

        results: list[ExtractionResult] = []
        answers_list = parsed.get("answers", [])

        answer_by_id: dict[str, dict[str, Any]] = {}
        for ans in answers_list:
            qid = ans.get("question_id", "")
            answer_by_id[qid] = ans

        for q in questions:
            ans = answer_by_id.get(q.question_id, {})
            raw_citations = self._citations_from_answer(ans)
            citations, source_pages, evidence_text = self._parse_citations(raw_citations)
            results.append(
                ExtractionResult(
                    question_id=q.question_id,
                    answer=ans.get("answer", "Not found"),
                    confidence=min(max(float(ans.get("confidence", 0.0)), 0.0), 1.0),
                    citations=citations,
                    source_pages=source_pages,
                    evidence_text=evidence_text,
                )
            )

        return results

    # ── Tier 2/3: Individual extraction ──────────────────────────────

    async def _extract_individual(
        self,
        question: ExtractionQuestion,
        context: str,
        *,
        system_prompt: str | None = None,
        cache_system: bool = False,
    ) -> ExtractionResult:
        """Extract answer for a single Tier 2/3 question with reasoning."""
        if system_prompt:
            # Caching path: context is in system prompt
            prompt = _CACHED_INDIVIDUAL_PROMPT.format(question=question.question_text)
        else:
            # Legacy path
            prompt = INDIVIDUAL_EXTRACTION_PROMPT.format(
                context=context[:8000],
                question=question.question_text,
            )

        response = await self._client.complete(
            prompt,
            system_prompt=system_prompt,
            cache_system=cache_system,
        )
        parsed = self._client.extract_json(response)

        raw_citations = self._citations_from_answer(parsed)
        citations, source_pages, evidence_text = self._parse_citations(raw_citations)

        return ExtractionResult(
            question_id=question.question_id,
            answer=parsed.get("answer", "Not found"),
            confidence=min(max(float(parsed.get("confidence", 0.0)), 0.0), 1.0),
            citations=citations,
            source_pages=source_pages,
            evidence_text=evidence_text,
        )
