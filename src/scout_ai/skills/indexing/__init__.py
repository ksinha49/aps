"""Indexing skills: detect TOC, process TOC, verify, split, enrich, build."""

from __future__ import annotations

from scout_ai.skills.indexing.build_index import build_index
from scout_ai.skills.indexing.detect_toc import detect_toc
from scout_ai.skills.indexing.enrich_nodes import enrich_nodes
from scout_ai.skills.indexing.process_toc import process_toc
from scout_ai.skills.indexing.split_nodes import split_large_nodes
from scout_ai.skills.indexing.verify_toc import fix_incorrect_toc, verify_toc

__all__ = [
    "build_index",
    "detect_toc",
    "enrich_nodes",
    "process_toc",
    "split_large_nodes",
    "verify_toc",
    "fix_incorrect_toc",
]
