"""Tests for prefix stabilization — deterministic context ordering."""

from __future__ import annotations

import json

from scout_ai.context.prefix.sort_strategies import (
    sort_by_doc_id_page,
    sort_by_page_number,
    sort_by_section_path,
)
from scout_ai.context.prefix.stabilizer import PrefixStabilizer


class TestSortByPageNumber:
    """sort_by_page_number should order by (start_index, node_id)."""

    def test_sorts_by_start_index(self) -> None:
        nodes = [
            {"start_index": 5, "node_id": "a"},
            {"start_index": 1, "node_id": "b"},
            {"start_index": 3, "node_id": "c"},
        ]
        result = sort_by_page_number(nodes)
        assert [n["start_index"] for n in result] == [1, 3, 5]

    def test_tiebreaker_on_node_id(self) -> None:
        nodes = [
            {"start_index": 1, "node_id": "z"},
            {"start_index": 1, "node_id": "a"},
        ]
        result = sort_by_page_number(nodes)
        assert [n["node_id"] for n in result] == ["a", "z"]

    def test_empty_list(self) -> None:
        assert sort_by_page_number([]) == []


class TestSortBySectionPath:
    """sort_by_section_path should order by (section_path, start_index)."""

    def test_sorts_by_section_path(self) -> None:
        nodes = [
            {"section_path": "B.1", "start_index": 1},
            {"section_path": "A.1", "start_index": 2},
            {"section_path": "A.1", "start_index": 1},
        ]
        result = sort_by_section_path(nodes)
        assert [(n["section_path"], n["start_index"]) for n in result] == [
            ("A.1", 1),
            ("A.1", 2),
            ("B.1", 1),
        ]


class TestSortByDocIdPage:
    """sort_by_doc_id_page should order by (doc_id, start_index)."""

    def test_sorts_by_doc_id_then_page(self) -> None:
        nodes = [
            {"doc_id": "doc2", "start_index": 1},
            {"doc_id": "doc1", "start_index": 3},
            {"doc_id": "doc1", "start_index": 1},
        ]
        result = sort_by_doc_id_page(nodes)
        assert [(n["doc_id"], n["start_index"]) for n in result] == [
            ("doc1", 1),
            ("doc1", 3),
            ("doc2", 1),
        ]


class TestPrefixStabilizer:
    """PrefixStabilizer should deterministically sort nodes."""

    def test_stabilize_page_number(self) -> None:
        s = PrefixStabilizer("page_number")
        nodes = [
            {"start_index": 3, "node_id": "a"},
            {"start_index": 1, "node_id": "b"},
        ]
        result = s.stabilize(nodes)
        assert result[0]["start_index"] == 1

    def test_deterministic_output(self) -> None:
        """Same input always produces same output."""
        s = PrefixStabilizer("page_number")
        nodes = [
            {"start_index": 5, "node_id": "x"},
            {"start_index": 2, "node_id": "y"},
            {"start_index": 2, "node_id": "a"},
        ]
        result1 = s.stabilize(nodes)
        result2 = s.stabilize(nodes)
        assert result1 == result2

    def test_stabilize_json_deterministic(self) -> None:
        data1 = {"b": 2, "a": 1}
        data2 = {"a": 1, "b": 2}
        assert PrefixStabilizer.stabilize_json(data1) == PrefixStabilizer.stabilize_json(data2)

    def test_stabilize_json_sorted_keys(self) -> None:
        data = {"z": 1, "a": 2, "m": 3}
        result = json.loads(PrefixStabilizer.stabilize_json(data))
        assert list(result.keys()) == ["a", "m", "z"]

    def test_invalid_strategy_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Unknown sort strategy"):
            PrefixStabilizer("nonexistent")
