"""Flat-list-to-tree conversion and page grouping.

Internalized from ``post_processing()`` and ``page_list_to_group_text()``
in vanilla PageIndex's ``page_index.py``.
"""

from __future__ import annotations

import math
import logging
from typing import Any, Optional

from pageindex_rag.models import TreeNode
from pageindex_rag.providers.pageindex.tokenizer import TokenCounter

log = logging.getLogger(__name__)


class TreeBuilder:
    """Converts flat TOC lists into hierarchical ``TreeNode`` trees."""

    def __init__(self, token_counter: TokenCounter) -> None:
        self._tc = token_counter

    def build_tree(self, flat_toc: list[dict[str, Any]], total_pages: int) -> list[TreeNode]:
        """Convert a flat TOC (with ``structure``, ``title``, ``physical_index``,
        ``appear_start``) into a tree of ``TreeNode`` objects.

        Mirrors ``post_processing()`` → ``list_to_tree()``.
        """
        if not flat_toc:
            return []

        # Step 1: Assign start_index / end_index from physical_index
        enriched = self._assign_page_ranges(flat_toc, total_pages)

        # Step 2: Build parent-child hierarchy from dotted structure codes
        tree_dicts = self._list_to_tree(enriched)
        if not tree_dicts:
            # Fallback: treat as flat list of nodes
            return [self._dict_to_node(item) for item in enriched]

        return [self._dict_to_node(d) for d in tree_dicts]

    def group_pages(
        self,
        page_contents: list[str],
        token_lengths: list[int],
        max_tokens: int = 20_000,
        overlap_pages: int = 1,
    ) -> list[str]:
        """Split page texts into token-bounded groups with overlap.

        Mirrors ``page_list_to_group_text()``.
        """
        total_tokens = sum(token_lengths)
        if total_tokens <= max_tokens:
            return ["".join(page_contents)]

        expected_parts = math.ceil(total_tokens / max_tokens)
        avg_tokens = math.ceil(((total_tokens / expected_parts) + max_tokens) / 2)

        subsets: list[str] = []
        current_subset: list[str] = []
        current_count = 0

        for i, (content, tokens) in enumerate(zip(page_contents, token_lengths)):
            if current_count + tokens > avg_tokens and current_subset:
                subsets.append("".join(current_subset))
                overlap_start = max(i - overlap_pages, 0)
                current_subset = list(page_contents[overlap_start:i])
                current_count = sum(token_lengths[overlap_start:i])

            current_subset.append(content)
            current_count += tokens

        if current_subset:
            subsets.append("".join(current_subset))

        log.info(f"Divided pages into {len(subsets)} groups")
        return subsets

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _assign_page_ranges(
        flat_toc: list[dict[str, Any]], total_pages: int
    ) -> list[dict[str, Any]]:
        """Assign ``start_index`` and ``end_index`` from ``physical_index``.

        Uses ``appear_start`` to decide whether the next section's page is
        inclusive or exclusive of the current section.
        """
        for i, item in enumerate(flat_toc):
            item["start_index"] = item.get("physical_index")
            if i < len(flat_toc) - 1:
                next_item = flat_toc[i + 1]
                if next_item.get("appear_start") == "yes":
                    item["end_index"] = next_item["physical_index"] - 1
                else:
                    item["end_index"] = next_item["physical_index"]
            else:
                item["end_index"] = total_pages
        return flat_toc

    @staticmethod
    def _list_to_tree(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert flat list with dotted ``structure`` codes into nested dicts.

        Mirrors ``list_to_tree()`` from vanilla PageIndex.
        """
        nodes: dict[str, dict[str, Any]] = {}
        root_nodes: list[dict[str, Any]] = []

        for item in data:
            structure = item.get("structure")
            if structure is None:
                continue
            node: dict[str, Any] = {
                "title": item.get("title", ""),
                "start_index": item.get("start_index"),
                "end_index": item.get("end_index"),
                "children": [],
            }
            nodes[structure] = node

            parts = str(structure).split(".")
            parent_key = ".".join(parts[:-1]) if len(parts) > 1 else None

            if parent_key and parent_key in nodes:
                nodes[parent_key]["children"].append(node)
            else:
                root_nodes.append(node)

        return root_nodes

    def _dict_to_node(self, d: dict[str, Any]) -> TreeNode:
        """Recursively convert a nested dict into a ``TreeNode``."""
        children_raw = d.get("children") or d.get("nodes") or []
        children = [self._dict_to_node(c) for c in children_raw]
        return TreeNode(
            title=d.get("title", ""),
            start_index=d.get("start_index", 0),
            end_index=d.get("end_index", 0),
            children=children,
        )
