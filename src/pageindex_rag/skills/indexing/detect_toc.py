"""TOC detection skill — scans pages for a table of contents.

Extracted from ``indexer.py:_check_toc`` (lines 292-326) and
``_detect_page_numbers_in_toc`` (lines 328-338).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from pageindex_rag.skills.common.json_parser import extract_json

log = logging.getLogger(__name__)


@tool(context=True)
def detect_toc(page_texts: list[str], check_limit: int = 20, tool_context: ToolContext = None) -> str:  # type: ignore[assignment]
    """Scan the first N pages of a document for a table of contents.

    Uses the agent's LLM to check each page for TOC content,
    then determines if page numbers are present.

    Args:
        page_texts: List of page text content, ordered by page number (0-indexed).
        check_limit: Maximum number of pages to scan from the beginning.

    Returns:
        JSON with toc_content, toc_page_list, and page_index_given_in_toc.
    """
    # This is a planning tool — returns the prompt and data for the agent
    # to reason about. The agent calls the LLM via its own model.
    pages_to_check = page_texts[: min(check_limit, len(page_texts))]

    return json.dumps({
        "action": "detect_toc",
        "pages_to_check_count": len(pages_to_check),
        "instruction": (
            "For each page, determine if it contains a table of contents. "
            "A TOC lists section titles with page references. "
            "Medication lists, lab results, and problem lists are NOT tables of contents. "
            "Stop scanning after finding a non-TOC page following TOC pages. "
            "If TOC pages are found, check if they contain page numbers."
        ),
        "pages_preview": [
            {"page_index": i, "text_preview": p[:500]}
            for i, p in enumerate(pages_to_check)
        ],
    })


def check_toc_sync(
    pages: list[dict[str, Any]],
    toc_detect_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pure logic: assemble TOC detection result from per-page LLM responses.

    Args:
        pages: List of page dicts with 'text' and 'page_number'.
        toc_detect_results: Per-page results with 'toc_detected' bool.

    Returns:
        Dict with toc_content, toc_page_list, page_index_given_in_toc.
    """
    toc_page_list: list[int] = []
    last_was_toc = False

    for i, result in enumerate(toc_detect_results):
        is_toc = result.get("toc_detected") == "yes"
        if is_toc:
            toc_page_list.append(i)
            last_was_toc = True
        elif last_was_toc:
            break
        else:
            last_was_toc = False

    if not toc_page_list:
        return {
            "toc_content": None,
            "toc_page_list": [],
            "page_index_given_in_toc": "no",
        }

    toc_content = "".join(pages[i]["text"] for i in toc_page_list if i < len(pages))

    return {
        "toc_content": toc_content,
        "toc_page_list": toc_page_list,
        "page_index_given_in_toc": "pending",  # Agent determines this
    }
