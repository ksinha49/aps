"""TOC verification and fixing skill.

Extracted from ``indexer.py:_check_title_appearances``, ``_verify_toc``,
``_check_title_on_page``, ``_fix_incorrect_toc``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from scout_ai.providers.pageindex.tree_utils import (
    convert_physical_index_to_int,
)

log = logging.getLogger(__name__)


@tool(context=True)
def verify_toc(  # type: ignore[assignment]
    toc_items: list[dict[str, Any]],
    start_index: int = 1,
    tool_context: ToolContext = None,
) -> str:
    """Verify that TOC section titles appear on their assigned pages.

    For each TOC entry, checks if the title actually appears on the page
    at physical_index. Returns accuracy score and list of incorrect items.

    Args:
        toc_items: TOC entries with physical_index assigned.
        start_index: Starting page index.

    Returns:
        JSON with verification instructions and items to check.
    """
    pages_data = tool_context.invocation_state.get("pages", [])  # type: ignore[union-attr]

    items_to_check = []
    for idx, item in enumerate(toc_items):
        pi = item.get("physical_index")
        if pi is None:
            continue
        page_idx = pi - start_index
        if page_idx < 0 or page_idx >= len(pages_data):
            continue

        page_text = pages_data[page_idx].get("text", "") if isinstance(pages_data[page_idx], dict) else pages_data[page_idx].text
        items_to_check.append({
            "list_index": idx,
            "title": item["title"],
            "physical_index": pi,
            "page_text_preview": page_text[:500],
        })

    return json.dumps({
        "action": "verify_toc",
        "total_items": len(toc_items),
        "items_to_check": items_to_check,
        "instruction": (
            "For each item, check if the section title appears on the assigned page. "
            "Use fuzzy matching — ignore space inconsistencies. "
            "Return 'yes' or 'no' for each item."
        ),
    })


def compute_accuracy(
    toc_items: list[dict[str, Any]],
    verification_results: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    """Pure logic: compute TOC accuracy from verification results.

    Args:
        toc_items: The TOC being verified.
        verification_results: Per-item results with 'answer' = 'yes'/'no'.

    Returns:
        Tuple of (accuracy, list_of_incorrect_items).
    """
    correct = 0
    incorrect: list[dict[str, Any]] = []
    valid_count = 0

    for result in verification_results:
        if isinstance(result, Exception):
            continue
        valid_count += 1
        if result.get("answer") == "yes":
            correct += 1
        else:
            incorrect.append(result)

    accuracy = correct / valid_count if valid_count > 0 else 0.0
    return accuracy, incorrect


@tool(context=True)
def fix_incorrect_toc(  # type: ignore[assignment]
    toc_items: list[dict[str, Any]],
    incorrect: list[dict[str, Any]],
    start_index: int = 1,
    tool_context: ToolContext = None,
) -> str:
    """Fix TOC entries whose titles don't appear on their assigned pages.

    For each incorrect entry, searches a range of pages between the
    previous and next correct entries to find the actual page.

    Args:
        toc_items: Full TOC list.
        incorrect: Items flagged as incorrect by verify_toc.
        start_index: Starting page index.

    Returns:
        JSON with fix instructions and search ranges.
    """
    pages_data = tool_context.invocation_state.get("pages", [])  # type: ignore[union-attr]
    total_pages = len(pages_data)

    fix_tasks = []
    for item in incorrect:
        idx = item.get("list_index")
        if idx is None or idx < 0 or idx >= len(toc_items):
            continue

        # Find search range
        prev_pi = start_index
        for j in range(idx - 1, -1, -1):
            if toc_items[j].get("physical_index") is not None:
                prev_pi = toc_items[j]["physical_index"]
                break

        next_pi = total_pages + start_index - 1
        for j in range(idx + 1, len(toc_items)):
            if toc_items[j].get("physical_index") is not None:
                next_pi = toc_items[j]["physical_index"]
                break

        fix_tasks.append({
            "list_index": idx,
            "title": item.get("title", toc_items[idx].get("title", "")),
            "search_range_start": prev_pi,
            "search_range_end": next_pi,
        })

    return json.dumps({
        "action": "fix_incorrect_toc",
        "fix_tasks": fix_tasks,
        "instruction": (
            "For each incorrect entry, search the specified page range "
            "to find where the section title actually starts. "
            "Return the correct physical_index for each."
        ),
    })


def apply_fixes(
    toc_items: list[dict[str, Any]],
    fix_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pure logic: apply fix results to TOC items.

    Args:
        toc_items: The TOC being fixed.
        fix_results: Per-item results with 'list_index' and 'physical_index'.

    Returns:
        Updated TOC items.
    """
    for fix in fix_results:
        idx = fix.get("list_index")
        new_pi = fix.get("physical_index")
        if idx is not None and new_pi is not None and 0 <= idx < len(toc_items):
            resolved = convert_physical_index_to_int(new_pi)
            if isinstance(resolved, int):
                toc_items[idx]["physical_index"] = resolved

    return toc_items
