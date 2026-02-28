"""Integration tests for the full extraction pipeline."""

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
    PageContent,
    TreeNode,
)
from pageindex_rag.providers.pageindex.chat import PageIndexChat
from pageindex_rag.providers.pageindex.client import LLMClient
from pageindex_rag.providers.pageindex.retrieval import PageIndexRetrieval
from pageindex_rag.services.extraction_service import ExtractionService


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
    tree = [
        TreeNode(
            node_id="0000",
            title="Face Sheet",
            start_index=1,
            end_index=1,
            text="Patient: John Doe, DOB: 01/15/1960, SSN: XXX-XX-1234",
            content_type=MedicalSectionType.FACE_SHEET,
        ),
        TreeNode(
            node_id="0001",
            title="Lab Report",
            start_index=5,
            end_index=5,
            text="CBC: WBC 7.2, RBC 4.8, Hgb 14.2. CMP: Glucose 95.",
            content_type=MedicalSectionType.LAB_REPORT,
        ),
    ]
    return DocumentIndex(
        doc_id="test-pipeline",
        doc_name="Pipeline Test",
        total_pages=10,
        tree=tree,
    )


@pytest.fixture
def pipeline():
    settings = PageIndexSettings(
        llm_base_url="http://test-llm:4000/v1",
        llm_api_key="test-key",
        llm_model="test-model",
    )
    client = LLMClient(settings)
    retrieval = PageIndexRetrieval(settings, client)
    chat = PageIndexChat(settings, client)
    return ExtractionService(retrieval, chat)


@pytest.mark.asyncio
class TestExtractionPipeline:
    async def test_full_pipeline_with_citations(self, pipeline, test_index):
        """New-format LLM response with structured citations."""
        questions = [
            ExtractionQuestion(
                question_id="q1",
                category=ExtractionCategory.DEMOGRAPHICS,
                question_text="What is the patient name?",
                tier=1,
            ),
            ExtractionQuestion(
                question_id="q2",
                category=ExtractionCategory.LAB_RESULTS,
                question_text="What is the WBC count?",
                tier=1,
            ),
        ]

        retrieval_resp = json.dumps({
            "reasoning": "Found in face sheet",
            "node_ids": ["0000"],
        })
        extraction_resp = json.dumps({
            "answers": [
                {
                    "question_id": "q1",
                    "answer": "John Doe",
                    "confidence": 0.95,
                    "citations": [
                        {
                            "page_number": 1,
                            "section_title": "Face Sheet",
                            "section_type": "face_sheet",
                            "verbatim_quote": "Patient: John Doe",
                        }
                    ],
                }
            ]
        })
        lab_retrieval_resp = json.dumps({
            "reasoning": "Found in lab report",
            "node_ids": ["0001"],
        })
        lab_extraction_resp = json.dumps({
            "answers": [
                {
                    "question_id": "q2",
                    "answer": "7.2",
                    "confidence": 0.9,
                    "citations": [
                        {
                            "page_number": 5,
                            "section_title": "Lab Report",
                            "section_type": "lab_report",
                            "verbatim_quote": "WBC 7.2",
                        }
                    ],
                }
            ]
        })

        async def _mock_acomp(**kwargs: Any) -> MagicMock:
            messages = kwargs.get("messages", [])
            prompt = messages[-1]["content"] if messages else ""

            if "demographics" in prompt.lower() or "identification" in prompt.lower():
                return _litellm_response(retrieval_resp)
            elif "lab" in prompt.lower() and "node_id" not in prompt:
                return _litellm_response(lab_retrieval_resp)
            elif "patient" in prompt.lower() and "name" in prompt.lower():
                return _litellm_response(extraction_resp)
            elif "wbc" in prompt.lower():
                return _litellm_response(lab_extraction_resp)
            else:
                return _litellm_response(extraction_resp)

        with patch("litellm.acompletion", side_effect=_mock_acomp):
            results = await pipeline.extract(test_index, questions)

        assert len(results) == 2

        for batch_result in results:
            assert batch_result.category in (
                ExtractionCategory.DEMOGRAPHICS,
                ExtractionCategory.LAB_RESULTS,
            )
            assert batch_result.retrieval is not None
            for ext in batch_result.extractions:
                assert len(ext.citations) >= 1
                assert ext.citations[0].verbatim_quote != ""
                assert ext.source_pages == [ext.citations[0].page_number]
                assert ext.evidence_text == ext.citations[0].verbatim_quote

    async def test_old_format_fallback(self, pipeline, test_index):
        """Old-format LLM response (source_pages/evidence_text) still works."""
        questions = [
            ExtractionQuestion(
                question_id="q1",
                category=ExtractionCategory.DEMOGRAPHICS,
                question_text="What is the patient name?",
                tier=1,
            ),
        ]

        retrieval_resp = json.dumps({
            "reasoning": "Found in face sheet",
            "node_ids": ["0000"],
        })
        extraction_resp = json.dumps({
            "answers": [
                {
                    "question_id": "q1",
                    "answer": "John Doe",
                    "confidence": 0.95,
                    "source_pages": [1],
                    "evidence_text": "Patient: John Doe",
                }
            ]
        })

        async def _mock_acomp(**kwargs: Any) -> MagicMock:
            messages = kwargs.get("messages", [])
            prompt = messages[-1]["content"] if messages else ""
            if "node_id" in prompt:
                return _litellm_response(retrieval_resp)
            return _litellm_response(extraction_resp)

        with patch("litellm.acompletion", side_effect=_mock_acomp):
            results = await pipeline.extract(test_index, questions)

        assert len(results) == 1
        ext = results[0].extractions[0]
        assert ext.answer == "John Doe"
        assert len(ext.citations) == 1
        assert ext.citations[0].page_number == 1
        assert ext.citations[0].verbatim_quote == "Patient: John Doe"
        assert ext.source_pages == [1]

    async def test_tiered_extraction_with_citations(self, pipeline, test_index):
        """Tier 2/3 questions get individual prompts with structured citations."""
        questions = [
            ExtractionQuestion(
                question_id="q1",
                category=ExtractionCategory.PROGNOSIS,
                question_text="What is the expected recovery timeline?",
                tier=2,
            ),
        ]

        retrieval_resp = json.dumps({
            "reasoning": "Prognosis info",
            "node_ids": ["0000"],
        })
        extraction_resp = json.dumps({
            "reasoning": "Cross-referencing treatment plan and progress notes",
            "answer": "6-8 weeks recovery expected",
            "confidence": 0.7,
            "citations": [
                {
                    "page_number": 1,
                    "section_title": "Face Sheet",
                    "section_type": "face_sheet",
                    "verbatim_quote": "Recovery timeline discussed",
                }
            ],
        })

        async def _mock_acomp(**kwargs: Any) -> MagicMock:
            messages = kwargs.get("messages", [])
            prompt = messages[-1]["content"] if messages else ""
            if "node_id" in prompt:
                return _litellm_response(retrieval_resp)
            return _litellm_response(extraction_resp)

        with patch("litellm.acompletion", side_effect=_mock_acomp):
            results = await pipeline.extract(test_index, questions)

        assert len(results) == 1
        assert results[0].category == ExtractionCategory.PROGNOSIS
        ext = results[0].extractions[0]
        assert ext.answer == "6-8 weeks recovery expected"
        assert len(ext.citations) == 1
        assert ext.citations[0].page_number == 1
        assert ext.citations[0].section_title == "Face Sheet"
        assert ext.source_pages == [1]

    async def test_extract_with_pages_param(self, pipeline, test_index):
        """Pages param enables per-page markers in context."""
        questions = [
            ExtractionQuestion(
                question_id="q1",
                category=ExtractionCategory.DEMOGRAPHICS,
                question_text="What is the patient name?",
                tier=1,
            ),
        ]

        pages = [
            PageContent(page_number=1, text="Patient: John Doe, DOB: 01/15/1960"),
        ]

        retrieval_resp = json.dumps({
            "reasoning": "Found in face sheet",
            "node_ids": ["0000"],
        })
        extraction_resp = json.dumps({
            "answers": [
                {
                    "question_id": "q1",
                    "answer": "John Doe",
                    "confidence": 0.95,
                    "citations": [
                        {
                            "page_number": 1,
                            "section_title": "Face Sheet",
                            "section_type": "face_sheet",
                            "verbatim_quote": "Patient: John Doe",
                        }
                    ],
                }
            ]
        })

        captured_prompts: list[str] = []

        async def _mock_acomp(**kwargs: Any) -> MagicMock:
            messages = kwargs.get("messages", [])
            prompt = messages[-1]["content"] if messages else ""
            captured_prompts.append(prompt)
            if "node_id" in prompt:
                return _litellm_response(retrieval_resp)
            return _litellm_response(extraction_resp)

        with patch("litellm.acompletion", side_effect=_mock_acomp):
            results = await pipeline.extract(test_index, questions, pages=pages)

        assert len(results) == 1
        extraction_prompt = [
            p for p in captured_prompts
            if "citations" in p.lower() and "node_ids" not in p
        ]
        assert extraction_prompt, "Expected an extraction prompt with citation instructions"
        assert "[Section:" in extraction_prompt[0]
        assert "[Page" in extraction_prompt[0]
