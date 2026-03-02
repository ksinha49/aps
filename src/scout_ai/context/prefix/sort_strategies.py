"""Sort strategies for deterministic node ordering.

Each strategy produces a stable sort key so that identical node sets
always produce identical context strings — maximizing prompt cache hits.
"""

from __future__ import annotations

from typing import Any


def sort_by_page_number(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort nodes by (start_index, node_id) for page-order stability."""
    return sorted(
        nodes,
        key=lambda n: (n.get("start_index", 0), n.get("node_id", "")),
    )


def sort_by_section_path(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort nodes by (section_path, start_index) for tree-order stability."""
    return sorted(
        nodes,
        key=lambda n: (n.get("section_path", ""), n.get("start_index", 0)),
    )


def sort_by_doc_id_page(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort nodes by (doc_id, start_index) for multi-document stability."""
    return sorted(
        nodes,
        key=lambda n: (n.get("doc_id", ""), n.get("start_index", 0)),
    )
