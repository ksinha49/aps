"""Individual extraction skill — Tier 2/3 extraction with reasoning chains.

Replaces ``providers/pageindex/chat.py`` Tier 2/3 logic. Each complex
question gets its own prompt with step-by-step reasoning.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from scout_ai.models import Citation, ExtractionResult

log = logging.getLogger(__name__)


@tool(context=True)
def extract_individual(  # type: ignore[assignment]
    question_id: str,
    question_text: str,
    context: str,
    tool_context: ToolContext = None,
) -> str:
    """Extract an answer for a single Tier 2/3 question with reasoning chain.

    Tier 2/3 questions require careful analysis and cross-referencing
    across multiple sections of the document.

    Args:
        question_id: Unique question identifier.
        question_text: The question to answer.
        context: Document context with [Page N] markers and section headers.

    Returns:
        JSON with individual extraction instructions for the agent.
    """
    settings = tool_context.invocation_state.get("settings")  # type: ignore[union-attr]
    max_context = 8000
    if settings:
        max_context = settings.extraction.max_context_chars

    return json.dumps({
        "action": "extract_individual",
        "question_id": question_id,
        "question_text": question_text,
        "context_length": len(context),
        "context_preview": context[:max_context],
        "instruction": (
            "Think step-by-step:\n"
            "1. Identify which parts of the context are relevant\n"
            "2. Cross-reference dates, providers, and findings\n"
            "3. Synthesize the answer\n\n"
            "Include citations with exact page numbers, section titles, "
            "and verbatim quotes. If not found, say 'Not found'."
        ),
    })


def parse_individual_result(
    question_id: str,
    llm_response: dict[str, Any],
) -> ExtractionResult:
    """Pure logic: parse individual extraction LLM response.

    Args:
        question_id: The question being answered.
        llm_response: Parsed LLM response.

    Returns:
        ExtractionResult object.
    """
    raw_citations = _citations_from_answer(llm_response)
    citations, source_pages, evidence_text = _parse_citations(raw_citations)

    return ExtractionResult(
        question_id=question_id,
        answer=llm_response.get("answer", "Not found"),
        confidence=min(max(float(llm_response.get("confidence", 0.0)), 0.0), 1.0),
        citations=citations,
        source_pages=source_pages,
        evidence_text=evidence_text,
    )


def _citations_from_answer(ans: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract raw citation list with old-format fallback."""
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
    """Parse citation dicts into models."""
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
