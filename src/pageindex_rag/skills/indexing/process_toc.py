"""TOC processing skill — three-mode TOC processing with fallback cascade.

Extracted from ``indexer.py:_meta_processor``, ``_process_no_toc``,
``_process_toc_with_page_numbers``, ``_process_toc_no_page_numbers``,
``_toc_transformer``, ``_extract_toc_indices``, ``_add_page_numbers_to_toc``.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from pageindex_rag.providers.pageindex.tree_utils import (
    add_preface_if_needed,
    convert_physical_index_to_int,
    get_text_of_pages,
    validate_physical_indices,
)
from pageindex_rag.skills.common.json_parser import extract_json

log = logging.getLogger(__name__)


@tool(context=True)
def process_toc(  # type: ignore[assignment]
    mode: str,
    toc_content: str = "",
    toc_page_list: list[int] = [],  # noqa: B006
    start_index: int = 1,
    tool_context: ToolContext = None,
) -> str:
    """Process document pages into a structured TOC using the specified mode.

    Three modes are available, tried in cascade on accuracy failures:
    - 'toc_with_pages': TOC has page numbers — transform + offset calculation
    - 'toc_no_pages': TOC exists but no page numbers — LLM maps sections
    - 'no_toc': No TOC found — LLM generates structure from content chunks

    Args:
        mode: Processing mode ('toc_with_pages', 'toc_no_pages', 'no_toc').
        toc_content: Raw TOC text (for modes with TOC).
        toc_page_list: Page indices where TOC was found.
        start_index: Starting page index (usually 1).

    Returns:
        JSON with the processing instruction and data for the agent.
    """
    pages_data = tool_context.invocation_state.get("pages", [])  # type: ignore[union-attr]
    settings = tool_context.invocation_state.get("settings")  # type: ignore[union-attr]

    max_group_tokens = 20_000
    if settings:
        max_group_tokens = settings.indexing.max_group_tokens

    return json.dumps({
        "action": "process_toc",
        "mode": mode,
        "toc_content_preview": toc_content[:1000] if toc_content else None,
        "toc_page_list": toc_page_list,
        "start_index": start_index,
        "total_pages": len(pages_data),
        "max_group_tokens": max_group_tokens,
        "instruction": _get_mode_instruction(mode),
    })


def _get_mode_instruction(mode: str) -> str:
    if mode == "toc_with_pages":
        return (
            "Transform the TOC into structured JSON, calculate page offset "
            "between logical and physical pages, and assign physical_index to each entry."
        )
    if mode == "toc_no_pages":
        return (
            "Transform the TOC into structured JSON, then scan document pages "
            "to find where each section physically starts."
        )
    return (
        "Divide pages into token-bounded groups. For each group, extract the "
        "hierarchical section structure with physical_index tags."
    )


def process_no_toc_sync(
    page_contents: list[str],
    group_texts: list[str],
    init_response: str,
    continue_responses: list[str],
) -> list[dict[str, Any]]:
    """Pure logic: assemble TOC from chunked LLM responses (Mode 3).

    Args:
        page_contents: Labeled page texts.
        group_texts: Token-bounded page groups.
        init_response: LLM response for first group.
        continue_responses: LLM responses for subsequent groups.

    Returns:
        List of TOC items with physical_index as integers.
    """
    toc = _parse_toc_json(init_response)

    for resp in continue_responses:
        additional = _parse_toc_json(resp)
        if isinstance(additional, list):
            toc.extend(additional)

    toc = convert_physical_index_to_int(toc)
    return toc


def process_toc_with_pages_sync(
    toc_json: list[dict[str, Any]],
    physical_indices: list[dict[str, Any]],
    start_page: int,
) -> list[dict[str, Any]]:
    """Pure logic: assign physical_index using page offset (Mode 1).

    Args:
        toc_json: Structured TOC entries with 'page' fields.
        physical_indices: LLM-identified physical page mappings.
        start_page: Physical page number after TOC pages.

    Returns:
        TOC items with physical_index assigned.
    """
    offset = _calculate_page_offset(toc_json, physical_indices, start_page)
    if offset is not None:
        for item in toc_json:
            if item.get("page") is not None and isinstance(item["page"], int):
                item["physical_index"] = item["page"] + offset
                del item["page"]

    return convert_physical_index_to_int(toc_json)


def process_toc_no_pages_sync(
    toc_json: list[dict[str, Any]],
    add_pages_responses: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Pure logic: fill physical_index from LLM responses (Mode 2).

    Args:
        toc_json: Structured TOC without page numbers.
        add_pages_responses: Per-group LLM responses with physical_index.

    Returns:
        TOC items with physical_index assigned.
    """
    toc_with_pages = copy.deepcopy(toc_json)
    for response_items in add_pages_responses:
        if isinstance(response_items, list):
            # Merge physical_index from response into toc
            by_title = {item.get("title"): item for item in response_items}
            for item in toc_with_pages:
                matched = by_title.get(item.get("title"))
                if matched and matched.get("physical_index") is not None:
                    item["physical_index"] = matched["physical_index"]

    return convert_physical_index_to_int(toc_with_pages)


def meta_processor_sync(
    toc_items: list[dict[str, Any]],
    total_pages: int,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    """Post-process TOC items: filter, validate, add preface.

    Args:
        toc_items: Raw TOC items from any mode.
        total_pages: Total number of document pages.
        start_index: Starting page index.

    Returns:
        Cleaned and validated TOC items.
    """
    toc_items = [i for i in toc_items if i.get("physical_index") is not None]
    toc_items = validate_physical_indices(toc_items, total_pages, start_index)
    toc_items = [i for i in toc_items if i.get("physical_index") is not None]
    toc_items = add_preface_if_needed(toc_items)
    return toc_items


def _parse_toc_json(response: str) -> list[dict[str, Any]]:
    """Parse a TOC JSON response."""
    result = extract_json(response)
    if isinstance(result, list):
        return result
    if isinstance(result, dict) and "table_of_contents" in result:
        return result["table_of_contents"]
    return []


def _calculate_page_offset(
    toc_with_pages: list[dict[str, Any]],
    physical_indices: list[dict[str, Any]],
    start_page_index: int,
) -> int | None:
    """Calculate offset between logical and physical page numbers."""
    from collections import Counter

    pi_by_title = {item.get("title"): item.get("physical_index") for item in physical_indices}
    pairs: list[tuple[int, int]] = []

    for item in toc_with_pages:
        title = item.get("title")
        page = item.get("page")
        pi = pi_by_title.get(title)
        if page is not None and pi is not None and isinstance(page, int) and isinstance(pi, int):
            if pi >= start_page_index:
                pairs.append((pi, page))

    if not pairs:
        return None

    diffs = [pi - page for pi, page in pairs]
    counts = Counter(diffs)
    return counts.most_common(1)[0][0]
