"""Integration tests for ScoutIndexer with mocked LLM via litellm."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scout_ai.config import ScoutSettings
from scout_ai.providers.pageindex.client import LLMClient
from scout_ai.providers.pageindex.indexer import ScoutIndexer


def _litellm_response(content: str) -> MagicMock:
    """Build a mock LiteLLM response object."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def mock_settings():
    return ScoutSettings(
        llm_base_url="http://test-llm:4000/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        tokenizer_method="approximate",
        enable_node_summaries=False,
        enable_section_classification=True,
        enable_doc_description=False,
        max_pages_per_node=3,
        max_tokens_per_node=500,
    )


@pytest.fixture
def mock_client(mock_settings):
    return LLMClient(mock_settings)


@pytest.fixture
def indexer(mock_settings, mock_client):
    return ScoutIndexer(mock_settings, mock_client)


@pytest.mark.asyncio
class TestIndexerWithHeuristics:
    """Test indexing when medical heuristics find enough sections."""

    async def test_build_index_heuristic_path(self, indexer, sample_pages):
        """When heuristics detect >= 3 sections, should skip LLM TOC detection."""

        async def _mock_acomp(**kwargs: Any) -> MagicMock:
            return _litellm_response(
                json.dumps({"thinking": "title found", "start_begin": "yes"})
            )

        with patch("litellm.acompletion", side_effect=_mock_acomp):
            result = await indexer.build_index(
                pages=sample_pages,
                doc_id="test-001",
                doc_name="Test APS",
            )

        assert result.doc_id == "test-001"
        assert result.doc_name == "Test APS"
        assert result.total_pages == 10
        assert len(result.tree) >= 1
        # Nodes should have IDs assigned
        assert result.tree[0].node_id != ""


@pytest.mark.asyncio
class TestIndexerNoToc:
    """Test indexing when no TOC is found (mode 3 / LLM generates structure)."""

    async def test_build_index_no_toc_fallback(self, mock_settings, sample_pages):
        """When heuristics find < 3 sections and no TOC, uses LLM generation."""
        # Disable medical classification to force LLM path
        mock_settings.enable_section_classification = False
        client = LLMClient(mock_settings)
        indexer = ScoutIndexer(mock_settings, client)

        toc_response = json.dumps({"thinking": "no toc", "toc_detected": "no"})
        generated_toc = json.dumps([
            {"structure": "1", "title": "Face Sheet", "physical_index": "<physical_index_1>"},
            {"structure": "2", "title": "History", "physical_index": "<physical_index_2>"},
            {"structure": "3", "title": "Progress Notes", "physical_index": "<physical_index_3>"},
            {"structure": "4", "title": "Labs", "physical_index": "<physical_index_5>"},
            {"structure": "5", "title": "Imaging", "physical_index": "<physical_index_6>"},
            {"structure": "6", "title": "Medications", "physical_index": "<physical_index_7>"},
            {"structure": "7", "title": "Surgery", "physical_index": "<physical_index_9>"},
            {"structure": "8", "title": "Discharge", "physical_index": "<physical_index_10>"},
        ])
        verify_response = json.dumps({"thinking": "found", "answer": "yes"})
        start_response = json.dumps({"thinking": "starts here", "start_begin": "yes"})

        call_count = 0

        async def _mock_acomp(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            messages = kwargs.get("messages", [])
            prompt = messages[-1]["content"] if messages else ""

            if "table of content" in prompt.lower() or "toc_detected" in prompt.lower():
                return _litellm_response(toc_response)
            elif "hierarchical tree structure" in prompt.lower() or "extract" in prompt.lower():
                return _litellm_response(generated_toc)
            elif "start_begin" in prompt.lower():
                return _litellm_response(start_response)
            elif "answer" in prompt.lower():
                return _litellm_response(verify_response)
            else:
                return _litellm_response(verify_response)

        with patch("litellm.acompletion", side_effect=_mock_acomp):
            result = await indexer.build_index(
                pages=sample_pages,
                doc_id="test-002",
                doc_name="No TOC APS",
            )

        assert result.doc_id == "test-002"
        assert result.total_pages == 10
        assert len(result.tree) >= 1
        assert call_count > 0
