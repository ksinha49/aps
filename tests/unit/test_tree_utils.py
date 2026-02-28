"""Unit tests for tree_utils pure functions."""

import pytest

from pageindex_rag.models import MedicalSectionType, PageContent, TreeNode
from pageindex_rag.providers.pageindex.tree_utils import (
    add_node_text,
    add_preface_if_needed,
    convert_physical_index_to_int,
    create_node_mapping,
    find_node_by_id,
    flatten_nodes,
    get_leaf_nodes,
    get_source_pages,
    get_text_of_pages,
    is_leaf_node,
    remove_fields,
    tree_to_dict,
    tree_to_toc_string,
    validate_physical_indices,
    write_node_ids,
)


class TestWriteNodeIds:
    def test_single_node(self):
        nodes = [TreeNode(title="A", start_index=1, end_index=5)]
        write_node_ids(nodes)
        assert nodes[0].node_id == "0000"

    def test_nested_tree(self, sample_tree):
        write_node_ids(sample_tree)
        assert sample_tree[0].node_id == "0000"
        assert sample_tree[0].children[0].node_id == "0001"
        assert sample_tree[0].children[1].node_id == "0002"

    def test_sequential_ids(self):
        nodes = [
            TreeNode(title="A", start_index=1, end_index=3),
            TreeNode(title="B", start_index=4, end_index=6),
        ]
        next_id = write_node_ids(nodes)
        assert nodes[0].node_id == "0000"
        assert nodes[1].node_id == "0001"
        assert next_id == 2


class TestAddNodeText:
    def test_populates_text(self, sample_tree, sample_pages):
        add_node_text(sample_tree, sample_pages)
        root = sample_tree[0]
        # Root covers pages 1-10 (all pages)
        assert "FACE SHEET" in root.text
        assert "DISCHARGE SUMMARY" in root.text

    def test_child_text_range(self, sample_tree, sample_pages):
        add_node_text(sample_tree, sample_pages)
        child = sample_tree[0].children[0]  # Progress Notes, pages 3-4
        assert "PROGRESS NOTE - 2024-01-15" in child.text
        assert "PROGRESS NOTE - 2024-02-10" in child.text
        assert "FACE SHEET" not in child.text


class TestFlattenNodes:
    def test_flat_list(self, sample_tree):
        flat = flatten_nodes(sample_tree)
        assert len(flat) == 3  # root + 2 children

    def test_preserves_order(self, sample_tree):
        flat = flatten_nodes(sample_tree)
        assert flat[0].title == "Patient Record"
        assert flat[1].title == "Progress Notes"
        assert flat[2].title == "Lab Report"


class TestGetLeafNodes:
    def test_returns_leaves(self, sample_tree):
        leaves = get_leaf_nodes(sample_tree)
        assert len(leaves) == 2
        titles = {l.title for l in leaves}
        assert titles == {"Progress Notes", "Lab Report"}


class TestCreateNodeMapping:
    def test_mapping(self, sample_tree):
        mapping = create_node_mapping(sample_tree)
        assert "0000" in mapping
        assert "0001" in mapping
        assert mapping["0002"].title == "Lab Report"


class TestFindNodeById:
    def test_found(self, sample_tree):
        node = find_node_by_id(sample_tree, "0001")
        assert node is not None
        assert node.title == "Progress Notes"

    def test_not_found(self, sample_tree):
        node = find_node_by_id(sample_tree, "9999")
        assert node is None


class TestIsLeafNode:
    def test_leaf(self, sample_tree):
        assert is_leaf_node(sample_tree, "0001") is True

    def test_non_leaf(self, sample_tree):
        assert is_leaf_node(sample_tree, "0000") is False


class TestGetSourcePages:
    def test_range(self, sample_tree):
        pages = get_source_pages(sample_tree)
        assert pages == list(range(1, 11))

    def test_child_range(self, sample_tree):
        children = sample_tree[0].children
        pages = get_source_pages([children[0]])  # pages 3-4
        assert pages == [3, 4]


class TestTreeToDict:
    def test_basic_serialization(self, sample_tree):
        result = tree_to_dict(sample_tree)
        assert len(result) == 1
        assert result[0]["title"] == "Patient Record"
        assert "children" in result[0]
        assert len(result[0]["children"]) == 2

    def test_excludes_text_by_default(self, sample_tree):
        sample_tree[0].text = "some text"
        result = tree_to_dict(sample_tree)
        assert "text" not in result[0]

    def test_includes_text_when_asked(self, sample_tree):
        sample_tree[0].text = "some text"
        result = tree_to_dict(sample_tree, include_text=True)
        assert result[0]["text"] == "some text"


class TestTreeToTocString:
    def test_rendering(self, sample_tree):
        toc = tree_to_toc_string(sample_tree)
        assert "Patient Record" in toc
        assert "  Progress Notes" in toc
        assert "  Lab Report" in toc


class TestValidatePhysicalIndices:
    def test_removes_overflows(self):
        items = [
            {"title": "A", "physical_index": 5},
            {"title": "B", "physical_index": 15},
        ]
        result = validate_physical_indices(items, total_pages=10)
        assert result[0]["physical_index"] == 5
        assert result[1]["physical_index"] is None

    def test_keeps_valid(self):
        items = [{"title": "A", "physical_index": 10}]
        result = validate_physical_indices(items, total_pages=10)
        assert result[0]["physical_index"] == 10


class TestAddPrefaceIfNeeded:
    def test_inserts_preface(self):
        items = [{"title": "Chapter 1", "physical_index": 3}]
        result = add_preface_if_needed(items)
        assert len(result) == 2
        assert result[0]["title"] == "Preface"
        assert result[0]["physical_index"] == 1

    def test_no_preface_needed(self):
        items = [{"title": "Chapter 1", "physical_index": 1}]
        result = add_preface_if_needed(items)
        assert len(result) == 1


class TestConvertPhysicalIndexToInt:
    def test_tag_format(self):
        items = [{"physical_index": "<physical_index_42>"}]
        result = convert_physical_index_to_int(items)
        assert result[0]["physical_index"] == 42

    def test_bare_format(self):
        items = [{"physical_index": "physical_index_7"}]
        result = convert_physical_index_to_int(items)
        assert result[0]["physical_index"] == 7

    def test_already_int(self):
        items = [{"physical_index": 10}]
        result = convert_physical_index_to_int(items)
        assert result[0]["physical_index"] == 10

    def test_string_value(self):
        result = convert_physical_index_to_int("<physical_index_99>")
        assert result == 99


class TestRemoveFields:
    def test_removes_nested(self):
        data = {"a": 1, "text": "remove", "nodes": [{"b": 2, "text": "also remove"}]}
        result = remove_fields(data, ["text"])
        assert "text" not in result
        assert "text" not in result["nodes"][0]


class TestGetTextOfPages:
    def test_range(self, sample_pages):
        text = get_text_of_pages(sample_pages, 1, 2)
        assert "FACE SHEET" in text
        assert "HISTORY AND PHYSICAL" in text
        assert "PROGRESS NOTE" not in text

    def test_with_labels(self, sample_pages):
        text = get_text_of_pages(sample_pages, 1, 1, with_labels=True)
        assert "<physical_index_1>" in text
