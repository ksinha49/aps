"""Tests for SynthesisPipeline — category filtering and summary generation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from pageindex_rag.config import PageIndexSettings
from pageindex_rag.models import (
    BatchExtractionResult,
    Citation,
    ExtractionCategory,
    ExtractionResult,
    RetrievalResult,
)
from pageindex_rag.providers.pageindex.client import LLMClient
from pageindex_rag.synthesis.models import SynthesisSection, UnderwriterSummary
from pageindex_rag.synthesis.pipeline import SynthesisPipeline


def _make_settings() -> PageIndexSettings:
    return PageIndexSettings(
        llm_base_url="http://localhost:4000/v1",
        llm_api_key="test-key",
        llm_model="test-model",
    )


def _make_batch_results() -> list[BatchExtractionResult]:
    return [
        BatchExtractionResult(
            category=ExtractionCategory.DEMOGRAPHICS,
            retrieval=RetrievalResult(query="demographics"),
            extractions=[
                ExtractionResult(
                    question_id="q1",
                    answer="John Doe",
                    confidence=0.95,
                    citations=[
                        Citation(page_number=1, verbatim_quote="Patient: John Doe")
                    ],
                ),
                ExtractionResult(
                    question_id="q2",
                    answer="01/15/1960",
                    confidence=0.9,
                    citations=[
                        Citation(page_number=1, verbatim_quote="DOB: 01/15/1960")
                    ],
                ),
            ],
        ),
        BatchExtractionResult(
            category=ExtractionCategory.LAB_RESULTS,
            retrieval=RetrievalResult(query="lab_results"),
            extractions=[
                ExtractionResult(
                    question_id="q3",
                    answer="7.2",
                    confidence=0.85,
                    citations=[
                        Citation(page_number=5, verbatim_quote="WBC 7.2")
                    ],
                ),
                ExtractionResult(
                    question_id="q4",
                    answer="Not found",
                    confidence=0.2,
                    citations=[],
                ),
            ],
        ),
    ]


_MOCK_SYNTHESIS_RESPONSE = """{
    "patient_demographics": "John Doe, DOB 01/15/1960",
    "sections": [
        {
            "title": "Patient Demographics",
            "content": "The patient is John Doe born on 01/15/1960.",
            "source_categories": ["demographics"],
            "key_findings": ["Patient identified as John Doe", "DOB: 01/15/1960"]
        },
        {
            "title": "Laboratory Results",
            "content": "WBC count of 7.2 within normal range.",
            "source_categories": ["lab_results"],
            "key_findings": ["WBC 7.2 - normal range"]
        }
    ],
    "risk_factors": ["Age over 60"],
    "overall_assessment": "Low risk profile based on available data."
}"""


class TestSynthesisPipelineCategoryFiltering:
    def test_prepare_category_summaries(self) -> None:
        client = LLMClient(_make_settings())
        pipeline = SynthesisPipeline(client)
        results = _make_batch_results()

        summaries = pipeline._prepare_category_summaries(results)

        assert len(summaries) == 2

        demo_summary = summaries[0]
        assert demo_summary["category"] == "demographics"
        assert demo_summary["question_count"] == 2
        assert demo_summary["high_confidence_count"] == 2  # Both >= 0.7
        assert len(demo_summary["answers"]) == 2

        lab_summary = summaries[1]
        assert lab_summary["category"] == "lab_results"
        assert lab_summary["question_count"] == 2
        assert lab_summary["high_confidence_count"] == 1  # q4 has 0.2 confidence
        assert len(lab_summary["answers"]) == 1  # Only q3 (high confidence)
        assert lab_summary["answers"][0]["question_id"] == "q3"

    def test_low_confidence_filtered_out(self) -> None:
        client = LLMClient(_make_settings())
        pipeline = SynthesisPipeline(client)

        results = [
            BatchExtractionResult(
                category=ExtractionCategory.ALLERGIES,
                retrieval=RetrievalResult(query="allergies"),
                extractions=[
                    ExtractionResult(
                        question_id="q10",
                        answer="Maybe penicillin",
                        confidence=0.3,
                    ),
                    ExtractionResult(
                        question_id="q11",
                        answer="Not found",
                        confidence=0.1,
                    ),
                ],
            ),
        ]

        summaries = pipeline._prepare_category_summaries(results)
        assert summaries[0]["high_confidence_count"] == 0
        assert len(summaries[0]["answers"]) == 0


class TestSynthesisPipelineGeneration:
    @pytest.mark.asyncio
    async def test_synthesize_produces_summary(self) -> None:
        """Full synthesis pipeline produces a structured UnderwriterSummary."""
        client = LLMClient(_make_settings())
        pipeline = SynthesisPipeline(client, cache_enabled=False)

        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = _MOCK_SYNTHESIS_RESPONSE
            summary = await pipeline.synthesize(
                _make_batch_results(),
                document_metadata={"doc_id": "test-doc"},
            )

        assert isinstance(summary, UnderwriterSummary)
        assert summary.document_id == "test-doc"
        assert summary.patient_demographics == "John Doe, DOB 01/15/1960"
        assert len(summary.sections) == 2
        assert summary.sections[0].title == "Patient Demographics"
        assert summary.sections[1].title == "Laboratory Results"
        assert "Age over 60" in summary.risk_factors
        assert summary.overall_assessment == "Low risk profile based on available data."
        assert summary.total_questions_answered == 4  # All questions
        assert summary.high_confidence_count == 3  # q1, q2, q3 have >= 0.7
        assert summary.generated_at != ""

    @pytest.mark.asyncio
    async def test_synthesize_with_caching(self) -> None:
        """When cache_enabled=True, system_prompt is passed to client."""
        client = LLMClient(_make_settings())
        pipeline = SynthesisPipeline(client, cache_enabled=True)

        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = _MOCK_SYNTHESIS_RESPONSE
            await pipeline.synthesize(_make_batch_results())

        # Verify system_prompt and cache_system were passed
        call_kwargs = mock_complete.call_args
        assert call_kwargs.kwargs["system_prompt"] is not None
        assert call_kwargs.kwargs["cache_system"] is True

    @pytest.mark.asyncio
    async def test_synthesize_without_caching(self) -> None:
        """When cache_enabled=False, system_prompt is None."""
        client = LLMClient(_make_settings())
        pipeline = SynthesisPipeline(client, cache_enabled=False)

        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = _MOCK_SYNTHESIS_RESPONSE
            await pipeline.synthesize(_make_batch_results())

        call_kwargs = mock_complete.call_args
        assert call_kwargs.kwargs["system_prompt"] is None
        assert call_kwargs.kwargs["cache_system"] is False

    @pytest.mark.asyncio
    async def test_synthesize_empty_results(self) -> None:
        """Synthesis handles empty extraction results gracefully."""
        client = LLMClient(_make_settings())
        pipeline = SynthesisPipeline(client, cache_enabled=False)

        empty_response = (
            '{"patient_demographics": "", "sections": [], '
            '"risk_factors": [], "overall_assessment": "Insufficient data."}'
        )
        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = empty_response
            summary = await pipeline.synthesize([], document_metadata={"doc_id": "empty"})

        assert summary.total_questions_answered == 0
        assert summary.high_confidence_count == 0
        assert len(summary.sections) == 0


class TestSynthesisModels:
    def test_synthesis_section_defaults(self) -> None:
        section = SynthesisSection(title="Test", content="Content")
        assert section.source_categories == []
        assert section.key_findings == []

    def test_underwriter_summary_defaults(self) -> None:
        summary = UnderwriterSummary(document_id="doc-1", patient_demographics="John")
        assert summary.sections == []
        assert summary.risk_factors == []
        assert summary.overall_assessment == ""
        assert summary.total_questions_answered == 0
