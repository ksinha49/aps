"""Batch extraction skill — Tier 1 batch answer extraction.

Replaces ``providers/pageindex/chat.py`` Tier 1 logic. Batches up to 20
questions per prompt for efficient extraction of simple lookup answers.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from scout_ai.models import Citation, ExtractionQuestion, ExtractionResult

log = logging.getLogger(__name__)


@tool(context=True)
def extract_batch(  # type: ignore[assignment]
    questions: list[dict[str, Any]],
    context: str,
    tool_context: ToolContext = None,
) -> str:
    """Extract answers for a batch of Tier 1 questions from medical context.

    Tier 1 questions are simple lookups (e.g., patient name, DOB, allergies)
    that can be efficiently answered in batches.

    Args:
        questions: List of question dicts with question_id and question_text.
        context: Document context with [Page N] markers and section headers.

    Returns:
        JSON with batch extraction instructions for the agent.
    """
    settings = tool_context.invocation_state.get("settings")  # type: ignore[union-attr]
    max_context = 8000
    if settings:
        max_context = settings.extraction.max_context_chars

    questions_text = "\n".join(
        f"- [{q.get('question_id', '')}] {q.get('question_text', '')}"
        for q in questions
    )

    return json.dumps({
        "action": "extract_batch",
        "question_count": len(questions),
        "context_length": len(context),
        "context_preview": context[:max_context],
        "questions_formatted": questions_text,
        "instruction": (
            "Answer each question based ONLY on the provided context. "
            "If the answer is not found, say 'Not found'. "
            "For each answer, include citations with exact page numbers "
            "(from [Page N] markers), section titles, and verbatim quotes."
        ),
    })


def parse_batch_results(
    questions: list[ExtractionQuestion],
    llm_response: dict[str, Any],
) -> list[ExtractionResult]:
    """Pure logic: parse batch extraction LLM response into ExtractionResults.

    Args:
        questions: Original questions being answered.
        llm_response: Parsed LLM response with 'answers' list.

    Returns:
        List of ExtractionResult objects.
    """
    answers_list = llm_response.get("answers", [])
    answer_by_id: dict[str, dict[str, Any]] = {}
    for ans in answers_list:
        qid = ans.get("question_id", "")
        answer_by_id[qid] = ans

    results: list[ExtractionResult] = []
    for q in questions:
        ans = answer_by_id.get(q.question_id, {})
        raw_citations = _citations_from_answer(ans)
        citations, source_pages, evidence_text = _parse_citations(raw_citations)
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


def _citations_from_answer(ans: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract raw citation list from an LLM answer, with old-format fallback."""
    raw = ans.get("citations", [])
    if raw:
        return raw
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
