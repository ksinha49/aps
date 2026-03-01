"""Unit tests for TreeBuilder."""

import pytest

from scout_ai.providers.pageindex.tokenizer import TokenCounter
from scout_ai.providers.pageindex.tree_builder import TreeBuilder


@pytest.fixture
def builder():
    tc = TokenCounter(method="approximate")
    return TreeBuilder(tc)


class TestBuildTree:
    def test_flat_toc(self, builder):
        flat = [
            {"structure": "1", "title": "Introduction", "physical_index": 1, "appear_start": "yes"},
            {"structure": "2", "title": "Methods", "physical_index": 5, "appear_start": "yes"},
            {"structure": "3", "title": "Results", "physical_index": 10, "appear_start": "yes"},
        ]
        tree = builder.build_tree(flat, total_pages=15)
        assert len(tree) == 3
        assert tree[0].title == "Introduction"
        assert tree[0].start_index == 1
        assert tree[0].end_index == 4  # next starts at 5, appear_start=yes, so 5-1=4
        assert tree[2].end_index == 15

    def test_nested_toc(self, builder):
        flat = [
            {"structure": "1", "title": "Chapter 1", "physical_index": 1, "appear_start": "yes"},
            {"structure": "1.1", "title": "Section 1.1", "physical_index": 2, "appear_start": "yes"},
            {"structure": "1.2", "title": "Section 1.2", "physical_index": 5, "appear_start": "yes"},
            {"structure": "2", "title": "Chapter 2", "physical_index": 8, "appear_start": "yes"},
        ]
        tree = builder.build_tree(flat, total_pages=10)
        assert len(tree) == 2  # Two root chapters
        assert len(tree[0].children) == 2  # Chapter 1 has 2 subsections

    def test_empty_toc(self, builder):
        tree = builder.build_tree([], total_pages=10)
        assert tree == []

    def test_appear_start_no(self, builder):
        """When next item's appear_start='no', end_index equals next section's page (inclusive)."""
        flat = [
            {"structure": "1", "title": "A", "physical_index": 1, "appear_start": "yes"},
            {"structure": "2", "title": "B", "physical_index": 5, "appear_start": "no"},
        ]
        tree = builder.build_tree(flat, total_pages=10)
        # appear_start on B is 'no', so A's end_index = B's physical_index (inclusive)
        assert tree[0].end_index == 5


class TestGroupPages:
    def test_single_group(self, builder):
        contents = ["Page 1 text. ", "Page 2 text. "]
        token_lengths = [100, 100]
        groups = builder.group_pages(contents, token_lengths, max_tokens=500)
        assert len(groups) == 1

    def test_multiple_groups(self, builder):
        contents = [f"Page {i} text. " for i in range(10)]
        token_lengths = [1000] * 10
        groups = builder.group_pages(contents, token_lengths, max_tokens=3000)
        assert len(groups) >= 2

    def test_overlap(self, builder):
        contents = ["A", "B", "C", "D", "E"]
        token_lengths = [500, 500, 500, 500, 500]
        groups = builder.group_pages(contents, token_lengths, max_tokens=800, overlap_pages=1)
        assert len(groups) >= 2
        # Due to overlap, content from boundary pages may appear in multiple groups
