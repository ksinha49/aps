"""Pure utility functions for tree manipulation.

Internalized from vanilla PageIndex ``utils.py`` — operates on ``TreeNode``
models instead of raw dicts.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from scout_ai.models import PageContent, TreeNode

# ── Node ID assignment ───────────────────────────────────────────────


def write_node_ids(nodes: list[TreeNode], start_id: int = 0) -> int:
    """Assign zero-padded sequential ``node_id`` values depth-first.

    Returns the next available id counter.
    """
    counter = start_id
    for node in nodes:
        node.node_id = str(counter).zfill(4)
        counter += 1
        counter = write_node_ids(node.children, counter)
    return counter


# ── Text population ──────────────────────────────────────────────────


def add_node_text(nodes: list[TreeNode], pages: list[PageContent]) -> None:
    """Populate ``node.text`` by concatenating page texts for the node's range."""
    for node in nodes:
        start = node.start_index
        end = node.end_index
        # Pages are 1-indexed; list is 0-indexed
        node.text = "".join(p.text for p in pages if start <= p.page_number <= end)
        if node.children:
            add_node_text(node.children, pages)


def add_node_text_with_labels(nodes: list[TreeNode], pages: list[PageContent]) -> None:
    """Like ``add_node_text`` but wraps each page with ``<physical_index_X>`` tags."""
    for node in nodes:
        parts: list[str] = []
        for p in pages:
            if node.start_index <= p.page_number <= node.end_index:
                parts.append(
                    f"<physical_index_{p.page_number}>\n{p.text}\n<physical_index_{p.page_number}>\n"
                )
        node.text = "".join(parts)
        if node.children:
            add_node_text_with_labels(node.children, pages)


# ── Node collection helpers ──────────────────────────────────────────


def flatten_nodes(nodes: list[TreeNode]) -> list[TreeNode]:
    """Return a flat list of all nodes (depth-first)."""
    result: list[TreeNode] = []
    for node in nodes:
        result.append(node)
        result.extend(flatten_nodes(node.children))
    return result


def get_leaf_nodes(nodes: list[TreeNode]) -> list[TreeNode]:
    """Return only leaf nodes (no children)."""
    leaves: list[TreeNode] = []
    for node in nodes:
        if not node.children:
            leaves.append(node)
        else:
            leaves.extend(get_leaf_nodes(node.children))
    return leaves


def create_node_mapping(nodes: list[TreeNode]) -> dict[str, TreeNode]:
    """Build a ``{node_id: node}`` lookup dict."""
    return {n.node_id: n for n in flatten_nodes(nodes)}


def find_node_by_id(nodes: list[TreeNode], node_id: str) -> Optional[TreeNode]:
    """Find a single node by ``node_id``."""
    for node in flatten_nodes(nodes):
        if node.node_id == node_id:
            return node
    return None


def is_leaf_node(nodes: list[TreeNode], node_id: str) -> bool:
    """Check if the node with *node_id* is a leaf."""
    node = find_node_by_id(nodes, node_id)
    return node is not None and len(node.children) == 0


# ── Source page extraction ───────────────────────────────────────────


def get_source_pages(nodes: list[TreeNode]) -> list[int]:
    """Collect unique sorted page numbers covered by *nodes*."""
    pages: set[int] = set()
    for node in nodes:
        pages.update(range(node.start_index, node.end_index + 1))
    return sorted(pages)


# ── Serialization ────────────────────────────────────────────────────


def tree_to_dict(nodes: list[TreeNode], include_text: bool = False) -> list[dict[str, Any]]:
    """Serialize a tree to plain dicts (for JSON output / LLM prompts)."""
    result: list[dict[str, Any]] = []
    for node in nodes:
        d: dict[str, Any] = {
            "node_id": node.node_id,
            "title": node.title,
            "start_index": node.start_index,
            "end_index": node.end_index,
        }
        if node.summary:
            d["summary"] = node.summary
        if node.content_type.value != "unknown":
            d["content_type"] = node.content_type.value
        if include_text and node.text:
            d["text"] = node.text
        if node.children:
            d["children"] = tree_to_dict(node.children, include_text=include_text)
        result.append(d)
    return result


def tree_to_toc_string(nodes: list[TreeNode], indent: int = 0) -> str:
    """Render a human-readable TOC string."""
    lines: list[str] = []
    for node in nodes:
        prefix = "  " * indent
        page_info = f" (pp. {node.start_index}-{node.end_index})"
        lines.append(f"{prefix}{node.title}{page_info}")
        if node.children:
            lines.append(tree_to_toc_string(node.children, indent + 1))
    return "\n".join(lines)


# ── Physical index validation ────────────────────────────────────────


def validate_physical_indices(
    toc_items: list[dict[str, Any]],
    total_pages: int,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    """Remove TOC items whose ``physical_index`` exceeds document length.

    Mirrors ``validate_and_truncate_physical_indices`` from vanilla PageIndex.
    """
    max_allowed = total_pages + start_index - 1
    for item in toc_items:
        pi = item.get("physical_index")
        if pi is not None and pi > max_allowed:
            item["physical_index"] = None
    return toc_items


# ── Preface insertion ────────────────────────────────────────────────


def add_preface_if_needed(toc_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Insert a "Preface" entry if the first section doesn't start at page 1."""
    if not toc_items:
        return toc_items
    first_pi = toc_items[0].get("physical_index")
    if first_pi is not None and first_pi > 1:
        toc_items.insert(0, {"structure": "0", "title": "Preface", "physical_index": 1})
    return toc_items


# ── Physical index tag parsing ───────────────────────────────────────


def convert_physical_index_to_int(data: Any) -> Any:
    """Parse ``<physical_index_X>`` tags to int values.

    Handles both list-of-dicts and bare strings.
    """
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "physical_index" in item:
                item["physical_index"] = _parse_physical_tag(item["physical_index"])
        return data
    if isinstance(data, str):
        return _parse_physical_tag(data)
    return data


def _parse_physical_tag(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.search(r"(\d+)", value.split("physical_index")[-1] if "physical_index" in value else "")
        if m:
            return int(m.group(1))
    return None


# ── Misc helpers ─────────────────────────────────────────────────────


def remove_fields(data: Any, fields: list[str]) -> Any:
    """Recursively remove *fields* from nested dicts/lists."""
    if isinstance(data, dict):
        return {k: remove_fields(v, fields) for k, v in data.items() if k not in fields}
    if isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data


def get_text_of_pages(pages: list[PageContent], start: int, end: int, with_labels: bool = False) -> str:
    """Get concatenated text for pages in range [start, end] (1-indexed)."""
    parts: list[str] = []
    for p in pages:
        if start <= p.page_number <= end:
            if with_labels:
                parts.append(f"<physical_index_{p.page_number}>\n{p.text}\n<physical_index_{p.page_number}>\n")
            else:
                parts.append(p.text)
    return "".join(parts)
