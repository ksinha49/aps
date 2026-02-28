"""Integration test: cached extraction with 5 questions via mocked LiteLLM."""

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
from pageindex_rag.providers.pageindex.chat import PageIndexChat
from pageindex_rag.providers.pageindex.client import LLMClient
from pageindex_rag.providers.pageindex.retrieval import PageIndexRetrieval
from pageindex_rag.services.extraction_service import ExtractionService


def _mock_response(content: str) -> MagicMock:
    """Build a mock LiteLLM response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def settings() -> PageIndexSettings:
    return PageIndexSettings(
        llm_base_url="http://test-llm:4000/v1",
        llm_api_key="test-key",
        llm_model="anthropic/claude-sonnet-4-20250514",
    )


@pytest.fixture
def test_index() -> DocumentIndex:
    tree = [
        TreeNode(
            node_id="0000",
            title="Face Sheet",
            start_index=1,
            end_index=1,
            text="Patient: John Doe, DOB: 01/15/1960, Gender: Male",
            content_type=MedicalSectionType.FACE_SHEET,
        ),
        TreeNode(
            node_id="0001",
            title="Lab Report",
            start_index=5,
            end_index=5,
            text="CBC: WBC 7.2, RBC 4.8, Hgb 14.2. CMP: Glucose 95, BUN 18.",
            content_type=MedicalSectionType.LAB_REPORT,
        ),
    ]
    return DocumentIndex(
        doc_id="test-cached",
        doc_name="Cached Extraction Test",
        total_pages=10,
        tree=tree,
    )


@pytest.fixture
def questions() -> list[ExtractionQuestion]:
    return [
        ExtractionQuestion(
            question_id="q1",
            category=ExtractionCategory.DEMOGRAPHICS,
            question_text="What is the patient name?",
            tier=1,
        ),
        ExtractionQuestion(
            question_id="q2",
            category=ExtractionCategory.DEMOGRAPHICS,
            question_text="What is the patient DOB?",
            tier=1,
        ),
        ExtractionQuestion(
            question_id="q3",
            category=ExtractionCategory.DEMOGRAPHICS,
            question_text="What is the patient gender?",
            tier=1,
        ),
        ExtractionQuestion(
            question_id="q4",
            category=ExtractionCategory.LAB_RESULTS,
            question_text="What is the WBC count?",
            tier=1,
        ),
        ExtractionQuestion(
            question_id="q5",
            category=ExtractionCategory.LAB_RESULTS,
            question_text="What is the glucose level?",
            tier=1,
        ),
    ]


@pytest.mark.asyncio
class TestCachedExtraction:
    async def test_extraction_with_caching_enabled(
        self, settings: PageIndexSettings, test_index: DocumentIndex, questions: list[ExtractionQuestion]
    ) -> None:
        """Extraction with cache_enabled=True sends system_prompt with cache_control."""
        client = LLMClient(settings)
        retrieval = PageIndexRetrieval(settings, client)
        chat = PageIndexChat(settings, client, cache_enabled=True)
        service = ExtractionService(retrieval, chat)

        captured_calls: list[dict[str, Any]] = []

        async def _mock_acomp(**kwargs: Any) -> MagicMock:
            captured_calls.append(kwargs)
            messages = kwargs["messages"]
            user_msg = messages[-1]["content"]

            # Retrieval calls
            if "node_id" in user_msg.lower() or "search" in user_msg.lower():
                return _mock_response(json.dumps({
                    "reasoning": "Found relevant nodes",
                    "node_ids": ["0000", "0001"],
                }))

            # Extraction calls
            return _mock_response(json.dumps({
                "answers": [
                    {
                        "question_id": qid,
                        "answer": "Answer for " + qid,
                        "confidence": 0.9,
                        "citations": [{
                            "page_number": 1,
                            "section_title": "Face Sheet",
                            "section_type": "face_sheet",
                            "verbatim_quote": "Evidence for " + qid,
                        }],
                    }
                    for qid in ["q1", "q2", "q3", "q4", "q5"]
                ],
            }))

        with patch("litellm.acompletion", side_effect=_mock_acomp):
            results = await service.extract(test_index, questions)

        # Verify we got results
        assert len(results) >= 1
        all_extractions = [e for br in results for e in br.extractions]
        assert len(all_extractions) >= 1

        # Verify that extraction calls used system_prompt with cache_control
        extraction_calls = [
            c for c in captured_calls
            if len(c["messages"]) >= 2 and c["messages"][0].get("role") == "system"
        ]

        for call in extraction_calls:
            system_msg = call["messages"][0]
            assert system_msg["role"] == "system"
            assert isinstance(system_msg["content"], list)
            content_block = system_msg["content"][0]
            assert content_block["cache_control"] == {"type": "ephemeral"}
            assert "Document Context:" in content_block["text"]

    async def test_extraction_without_caching(
        self, settings: PageIndexSettings, test_index: DocumentIndex, questions: list[ExtractionQuestion]
    ) -> None:
        """Extraction with cache_enabled=False uses legacy single-message format."""
        client = LLMClient(settings)
        retrieval = PageIndexRetrieval(settings, client)
        chat = PageIndexChat(settings, client, cache_enabled=False)
        service = ExtractionService(retrieval, chat)

        captured_calls: list[dict[str, Any]] = []

        async def _mock_acomp(**kwargs: Any) -> MagicMock:
            captured_calls.append(kwargs)
            messages = kwargs["messages"]
            user_msg = messages[-1]["content"]

            if "node_id" in user_msg.lower() or "search" in user_msg.lower():
                return _mock_response(json.dumps({
                    "reasoning": "Found",
                    "node_ids": ["0000"],
                }))

            return _mock_response(json.dumps({
                "answers": [{
                    "question_id": "q1",
                    "answer": "John Doe",
                    "confidence": 0.95,
                    "citations": [{
                        "page_number": 1,
                        "section_title": "Face Sheet",
                        "section_type": "face_sheet",
                        "verbatim_quote": "Patient: John Doe",
                    }],
                }],
            }))

        with patch("litellm.acompletion", side_effect=_mock_acomp):
            results = await service.extract(test_index, questions)

        assert len(results) >= 1

        # Legacy path: no system messages with cache_control
        for call in captured_calls:
            messages = call["messages"]
            if messages[0].get("role") == "system":
                content_block = messages[0]["content"]
                if isinstance(content_block, list):
                    assert "cache_control" not in content_block[0]
