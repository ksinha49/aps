"""Tests for the prompt registry: configure, get_prompt, auto-configure, fallback."""

from __future__ import annotations

import pytest

from pageindex_rag.prompts.context import PromptContext
from pageindex_rag.prompts.registry import (
    configure,
    get_active_context,
    get_prompt,
    reset,
    set_active_context,
)


@pytest.fixture(autouse=True)
def _reset_registry() -> None:  # type: ignore[misc]
    """Reset registry state before each test."""
    reset()
    yield  # type: ignore[misc]
    reset()


class TestAutoConfigureFileBackend:
    """get_prompt auto-configures to file backend when configure() is never called."""

    def test_loads_aps_indexing_prompt(self) -> None:
        prompt = get_prompt("aps", "indexing", "TOC_DETECT_PROMPT")
        assert "table of contents" in prompt.lower()
        assert "{content}" in prompt

    def test_loads_aps_retrieval_prompt(self) -> None:
        prompt = get_prompt("aps", "retrieval", "TREE_SEARCH_PROMPT")
        assert "{tree_structure}" in prompt

    def test_loads_aps_extraction_prompt(self) -> None:
        prompt = get_prompt("aps", "extraction", "BATCH_EXTRACTION_PROMPT")
        assert "{context}" in prompt

    def test_loads_aps_classification_prompt(self) -> None:
        prompt = get_prompt("aps", "classification", "CLASSIFY_SECTION_PROMPT")
        assert "{title}" in prompt

    def test_loads_base_indexing_agent_prompt(self) -> None:
        prompt = get_prompt("base", "indexing_agent", "INDEXING_SYSTEM_PROMPT")
        assert "document structure" in prompt.lower()

    def test_loads_base_retrieval_agent_prompt(self) -> None:
        prompt = get_prompt("base", "retrieval_agent", "RETRIEVAL_SYSTEM_PROMPT")
        assert "retrieval" in prompt.lower()

    def test_loads_base_extraction_agent_prompt(self) -> None:
        prompt = get_prompt("base", "extraction_agent", "EXTRACTION_SYSTEM_PROMPT")
        assert "extraction" in prompt.lower()


class TestExplicitFileConfigure:
    """Explicit configure(backend='file') works the same as auto-configure."""

    def test_file_backend_works(self) -> None:
        configure(backend="file")
        prompt = get_prompt("aps", "indexing", "TOC_DETECT_PROMPT")
        assert "table of contents" in prompt.lower()


class TestKeyError:
    """Missing prompts raise KeyError."""

    def test_missing_prompt_name(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            get_prompt("aps", "indexing", "NONEXISTENT_PROMPT")

    def test_missing_module(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            get_prompt("bogus_domain", "bogus_category", "SOME_PROMPT")


class TestContextVar:
    """Per-request ContextVar is used when no explicit context is passed."""

    def test_set_and_get_active_context(self) -> None:
        ctx = PromptContext(lob="life", department="uw")
        set_active_context(ctx)
        assert get_active_context() == ctx

    def test_default_is_none(self) -> None:
        assert get_active_context() is None

    def test_context_used_in_file_backend(self) -> None:
        set_active_context(PromptContext(lob="life"))
        prompt = get_prompt("aps", "indexing", "TOC_DETECT_PROMPT")
        assert "table of contents" in prompt.lower()


class TestBackwardCompat:
    """Backward compatibility: aps/prompts.py __getattr__ returns same strings."""

    def test_aps_prompts_import(self) -> None:
        from pageindex_rag.aps.prompts import TOC_DETECT_PROMPT

        registry_prompt = get_prompt("aps", "indexing", "TOC_DETECT_PROMPT")
        assert TOC_DETECT_PROMPT == registry_prompt

    def test_aps_prompts_retrieval(self) -> None:
        from pageindex_rag.aps.prompts import TREE_SEARCH_PROMPT

        registry_prompt = get_prompt("aps", "retrieval", "TREE_SEARCH_PROMPT")
        assert TREE_SEARCH_PROMPT == registry_prompt

    def test_aps_prompts_extraction(self) -> None:
        from pageindex_rag.aps.prompts import BATCH_EXTRACTION_PROMPT

        registry_prompt = get_prompt("aps", "extraction", "BATCH_EXTRACTION_PROMPT")
        assert BATCH_EXTRACTION_PROMPT == registry_prompt

    def test_template_module_import(self) -> None:
        from pageindex_rag.prompts.templates.aps.indexing import GENERATE_TOC_INIT_PROMPT

        registry_prompt = get_prompt("aps", "indexing", "GENERATE_TOC_INIT_PROMPT")
        assert GENERATE_TOC_INIT_PROMPT == registry_prompt
