"""Build index skill — facade orchestrating the full indexing pipeline.

This is the top-level skill that an indexing agent uses to coordinate
detect_toc → process_toc → verify_toc → split_nodes → enrich_nodes.

The actual orchestration is handled by the Strands agent; this skill
provides the pipeline plan and invokes the pure-logic functions.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from scout_ai.core.config import TokenizerConfig
from scout_ai.models import DocumentIndex, PageContent
from scout_ai.providers.pageindex.tokenizer import TokenCounter
from scout_ai.providers.pageindex.tree_builder import TreeBuilder
from scout_ai.providers.pageindex.tree_utils import (
    add_node_text,
    write_node_ids,
)

log = logging.getLogger(__name__)


@tool(context=True)
def build_index(  # type: ignore[assignment]
    doc_id: str,
    doc_name: str,
    tool_context: ToolContext = None,
) -> str:
    """Build a hierarchical tree index from pre-OCR'd document pages.

    This is the top-level indexing operation. It expects the pages
    to be available in invocation_state['pages'] as a list of
    PageContent objects (or dicts with 'page_number' and 'text').

    The tool returns the pipeline steps the agent should follow.

    Args:
        doc_id: Unique document identifier.
        doc_name: Human-readable document name.

    Returns:
        JSON with the indexing pipeline plan and page metadata.
    """
    pages_raw = tool_context.invocation_state.get("pages", [])  # type: ignore[union-attr]
    settings = tool_context.invocation_state.get("settings")  # type: ignore[union-attr]

    # Deep copy to avoid mutating caller's data
    pages = _ensure_page_content(pages_raw)

    if not pages:
        return json.dumps({"error": "No pages provided"})

    # Populate token counts
    tok_cfg = settings.tokenizer if settings else TokenizerConfig()
    tc = TokenCounter(method=tok_cfg.method, model=tok_cfg.model)
    for p in pages:
        if p.token_count is None:
            p.token_count = tc.count(p.text)

    total_tokens = sum(p.token_count or 0 for p in pages)

    return json.dumps({
        "action": "build_index",
        "doc_id": doc_id,
        "doc_name": doc_name,
        "total_pages": len(pages),
        "total_tokens": total_tokens,
        "pipeline": [
            "1. detect_toc: Scan first 20 pages for a table of contents",
            "2. process_toc: Process TOC using the appropriate mode",
            "3. verify_toc: Verify section titles appear on assigned pages",
            "4. split_large_nodes: Split any nodes exceeding size thresholds",
            "5. enrich_nodes: Add summaries, classification, description",
            "6. Assemble final DocumentIndex",
        ],
        "page_summaries": [
            {
                "page_number": p.page_number,
                "token_count": p.token_count,
                "text_preview": p.text[:200],
            }
            for p in pages[:5]
        ],
    })


def assemble_index(
    doc_id: str,
    doc_name: str,
    pages: list[PageContent],
    toc_items: list[dict[str, Any]],
    doc_description: str = "",
    settings: Any = None,
) -> DocumentIndex:
    """Pure logic: build the final DocumentIndex from processed TOC items.

    This is the final assembly step after all agent-driven processing.

    Args:
        doc_id: Document identifier.
        doc_name: Document name.
        pages: Page content list.
        toc_items: Validated and verified TOC items.
        doc_description: Optional generated description.
        settings: Optional AppSettings for tokenizer config.

    Returns:
        Complete DocumentIndex ready for persistence.
    """
    tok_cfg = settings.tokenizer if settings else TokenizerConfig()
    tc = TokenCounter(method=tok_cfg.method, model=tok_cfg.model)
    builder = TreeBuilder(tc)

    # Build tree from validated TOC items
    valid = [i for i in toc_items if i.get("physical_index") is not None]
    tree = builder.build_tree(valid, len(pages))

    # Assign IDs and populate text
    write_node_ids(tree)
    add_node_text(tree, pages)

    return DocumentIndex(
        doc_id=doc_id,
        doc_name=doc_name,
        doc_description=doc_description,
        total_pages=len(pages),
        tree=tree,
        created_at=datetime.now(timezone.utc),
    )


def _ensure_page_content(pages_raw: list[Any]) -> list[PageContent]:
    """Convert raw page data (dicts or PageContent) to PageContent list."""
    pages: list[PageContent] = []
    for p in pages_raw:
        if isinstance(p, PageContent):
            pages.append(copy.deepcopy(p))
        elif isinstance(p, dict):
            pages.append(PageContent(**p))
        else:
            pages.append(copy.deepcopy(p))
    return pages
