"""Extraction completions with tiered prompting and optional Anthropic prompt caching.

Tier 1 (simple lookup, ~60-70% of questions): batch 20 questions per prompt.
Tier 2/3 (cross-reference / reasoning): individual prompts with reasoning chains.

When caching is enabled, document context is placed in a system message with
``cache_control`` so it's cached server-side across all extraction calls.
"""

from __future__ import annotations

import logging
from typing import Any

from scout_ai.config import ScoutSettings
from scout_ai.interfaces.chat import IChatProvider
from scout_ai.models import Citation, ExtractionQuestion, ExtractionResult
from scout_ai.providers.pageindex.client import LLMClient

log = logging.getLogger(__name__)

TIER1_BATCH_SIZE = 20

_DEFAULT_SYSTEM_TEMPLATE = (
    "You are extracting information from a document.\n"
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


class ScoutChat(IChatProvider):
    """Extraction chat provider with tiered prompting strategy."""

    def __init__(
        self,
        settings: ScoutSettings,
        client: LLMClient,
        *,
        cache_enabled: bool = False,
        batch_extraction_prompt: str | None = None,
        individual_extraction_prompt: str | None = None,
        system_template: str | None = None,
        domain: str = "aps",
    ) -> None:
        self._settings = settings
        self._client = client
        self._cache_enabled = cache_enabled
        self._batch_extraction_prompt = batch_extraction_prompt or ""
        self._individual_extraction_prompt = individual_extraction_prompt or ""
        self._system_template = system_template or _DEFAULT_SYSTEM_TEMPLATE
        self._domain = domain

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
            system_prompt = self._system_template.format(context=context[:8000])

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
        prompt = f"""Based on the following document context, answer the question.

Context:
{context}

Question: {query}

Provide a detailed answer based only on the provided context."""
        return await self._client.complete(prompt)

    # ── LLM output coercion helpers ──────────────────────────────────

    @staticmethod
    def _safe_answer(raw: Any) -> str:
        """Coerce an answer value to str. Handles list answers from LLMs."""
        if raw is None:
            return "Not found"
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            return ", ".join(str(item) for item in raw)
        return str(raw)

    @staticmethod
    def _safe_confidence(raw: Any) -> float:
        """Coerce a confidence value to float in [0, 1]."""
        try:
            return min(max(float(raw), 0.0), 1.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _safe_page_number(raw: Any) -> int:
        """Coerce a page_number value to int, handling malformed LLM output.

        Handles: 6, "6", "Page 6", "page 6", "p6", "p. 6", None, etc.
        """
        if raw is None:
            return 0
        if isinstance(raw, int):
            return raw
        s = str(raw).strip()
        # Strip common prefixes: "Page 6", "page 6", "p6", "p. 6"
        import re

        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else 0

    @staticmethod
    def _parse_citations(
        raw_citations: list[dict[str, Any]],
    ) -> tuple[list[Citation], list[int], str]:
        """Parse citation dicts into models and derive backward-compat fields."""
        citations: list[Citation] = []
        pages: set[int] = set()
        quotes: list[str] = []
        for c in raw_citations:
            if not isinstance(c, dict):
                continue
            cit = Citation(
                page_number=ScoutChat._safe_page_number(c.get("page_number", 0)),
                section_title=str(c.get("section_title", "") or ""),
                section_type=str(c.get("section_type", "") or ""),
                verbatim_quote=str(c.get("verbatim_quote", "") or ""),
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
        if not isinstance(ans, dict):
            return []
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

    @staticmethod
    def _format_question(q: ExtractionQuestion) -> str:
        """Format a question with type-specific hints for the LLM."""
        base = f"- [{q.question_id}] {q.question_text}"
        if q.expected_type == "boolean_with_detail":
            base += " (Answer Y/N first, then provide detail)"
        elif q.expected_type == "list":
            base += " (Answer as a structured list)"
        return base

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
            self._format_question(q) for q in questions
        )

        if system_prompt:
            # Caching path: context is in system prompt, user message has only questions
            prompt = _CACHED_BATCH_PROMPT.format(questions=questions_text)
        else:
            # Legacy path: context + questions in a single user message
            if not self._batch_extraction_prompt:
                from scout_ai.prompts.registry import get_prompt

                self._batch_extraction_prompt = get_prompt(self._domain, "extraction", "BATCH_EXTRACTION_PROMPT")
            prompt = self._batch_extraction_prompt.format(
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
                    answer=self._safe_answer(ans.get("answer", "Not found")),
                    confidence=self._safe_confidence(ans.get("confidence", 0.0)),
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
        formatted_q = self._format_question(question).lstrip("- ")
        if system_prompt:
            # Caching path: context is in system prompt
            prompt = _CACHED_INDIVIDUAL_PROMPT.format(question=formatted_q)
        else:
            # Legacy path
            if not self._individual_extraction_prompt:
                from scout_ai.prompts.registry import get_prompt

                self._individual_extraction_prompt = get_prompt(
                    self._domain, "extraction", "INDIVIDUAL_EXTRACTION_PROMPT",
                )
            prompt = self._individual_extraction_prompt.format(
                context=context[:8000],
                question=formatted_q,
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
            answer=self._safe_answer(parsed.get("answer", "Not found")),
            confidence=self._safe_confidence(parsed.get("confidence", 0.0)),
            citations=citations,
            source_pages=source_pages,
            evidence_text=evidence_text,
        )
