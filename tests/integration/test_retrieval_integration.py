"""Integration tests for retrieval with mocked LLM."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pageindex_rag.config import PageIndexSettings
from pageindex_rag.models import (
    DocumentIndex,
    ExtractionCategory,
    ExtractionQuestion,
    MedicalSectionType,
    TreeNode,
)
from pageindex_rag.providers.pageindex.client import LLMClient
from pageindex_rag.providers.pageindex.retrieval import PageIndexRetrieval


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
def test_index() -> DocumentIndex:
    """Build a test index with known structure."""
    tree = [
        TreeNode(
            node_id="0000",
            title="Face Sheet",
            start_index=1,
            end_index=1,
            text="Patient: John Doe, DOB: 01/15/1960",
            content_type=MedicalSectionType.FACE_SHEET,
            summary="Patient demographics",
        ),
        TreeNode(
            node_id="0001",
            title="Progress Notes",
            start_index=2,
            end_index=4,
            text="BP 130/85, HR 72. Chronic lumbar radiculopathy. Improvement with PT.",
            content_type=MedicalSectionType.PROGRESS_NOTE,
            summary="Clinical notes showing improvement",
        ),
        TreeNode(
            node_id="0002",
            title="Lab Report",
            start_index=5,
            end_index=5,
            text="CBC: WBC 7.2, RBC 4.8, Hgb 14.2. CMP: Glucose 95.",
            content_type=MedicalSectionType.LAB_REPORT,
            summary="Normal lab results",
        ),
    ]
    return DocumentIndex(
        doc_id="test-idx",
        doc_name="Test APS",
        total_pages=10,
        tree=tree,
    )


@pytest.fixture
def retrieval():
    settings = PageIndexSettings(
        llm_base_url="http://test-llm:4000/v1",
        llm_api_key="test-key",
        llm_model="test-model",
    )
    client = LLMClient(settings)
    return PageIndexRetrieval(settings, client)


@pytest.mark.asyncio
class TestSingleQueryRetrieval:
    async def test_retrieve_returns_matching_nodes(self, retrieval, test_index):
        mock_content = json.dumps({
            "reasoning": "Lab results contain blood work data",
            "node_ids": ["0002"],
        })

        async def _mock_acomp(**kwargs: Any) -> MagicMock:
            return _litellm_response(mock_content)

        with patch("litellm.acompletion", side_effect=_mock_acomp):
            result = await retrieval.retrieve(test_index, "What are the lab results?")

        assert result.query == "What are the lab results?"
        assert len(result.retrieved_nodes) == 1
        assert result.retrieved_nodes[0]["node_id"] == "0002"
        assert result.source_pages == [5]
        assert result.reasoning != ""

    async def test_retrieve_multiple_nodes(self, retrieval, test_index):
        mock_content = json.dumps({
            "reasoning": "BP in progress notes, demographics in face sheet",
            "node_ids": ["0000", "0001"],
        })

        async def _mock_acomp(**kwargs: Any) -> MagicMock:
            return _litellm_response(mock_content)

        with patch("litellm.acompletion", side_effect=_mock_acomp):
            result = await retrieval.retrieve(test_index, "blood pressure?")

        assert len(result.retrieved_nodes) == 2
        assert result.source_pages == [1, 2, 3, 4]


@pytest.mark.asyncio
class TestBatchRetrieval:
    async def test_batch_groups_by_category(self, retrieval, test_index):
        questions = [
            ExtractionQuestion(
                question_id="q1",
                category=ExtractionCategory.DEMOGRAPHICS,
                question_text="Patient name?",
            ),
            ExtractionQuestion(
                question_id="q2",
                category=ExtractionCategory.DEMOGRAPHICS,
                question_text="Patient DOB?",
            ),
            ExtractionQuestion(
                question_id="q3",
                category=ExtractionCategory.LAB_RESULTS,
                question_text="WBC count?",
            ),
        ]

        call_count = 0

        async def _mock_acomp(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            return _litellm_response(
                json.dumps({"reasoning": "test", "node_ids": ["0000"]})
            )

        with patch("litellm.acompletion", side_effect=_mock_acomp):
            results = await retrieval.batch_retrieve(test_index, questions)

        assert ExtractionCategory.DEMOGRAPHICS in results
        assert ExtractionCategory.LAB_RESULTS in results
        # One LLM call per category, not per question
        assert call_count == 2
