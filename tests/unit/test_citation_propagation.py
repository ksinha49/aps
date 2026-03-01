"""Tests for citation propagation in the synthesis pipeline."""

from __future__ import annotations

from scout_ai.config import ScoutSettings
from scout_ai.models import (
    BatchExtractionResult,
    Citation,
    ExtractionCategory,
    ExtractionResult,
    RetrievalResult,
)
from scout_ai.providers.pageindex.client import LLMClient
from scout_ai.synthesis.pipeline import SynthesisPipeline


def _make_settings() -> ScoutSettings:
    return ScoutSettings(
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
                        Citation(
                            page_number=1,
                            section_title="Face Sheet",
                            section_type="face_sheet",
                            verbatim_quote="Patient: John Doe, DOB 01/15/1960",
                        ),
                    ],
                ),
                ExtractionResult(
                    question_id="q2",
                    answer="Male",
                    confidence=0.9,
                    citations=[
                        Citation(
                            page_number=1,
                            section_title="Face Sheet",
                            section_type="face_sheet",
                            verbatim_quote="Gender: Male",
                        ),
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
                        Citation(
                            page_number=5,
                            section_title="Chemistry Panel",
                            section_type="lab_report",
                            verbatim_quote="HbA1c: 7.2%",
                        ),
                    ],
                ),
                ExtractionResult(
                    question_id="q4",
                    answer="Not found",
                    confidence=0.2,
                    citations=[
                        Citation(
                            page_number=10,
                            section_title="Unknown",
                            section_type="unknown",
                            verbatim_quote="No relevant data",
                        ),
                    ],
                ),
            ],
        ),
    ]


class TestBuildCitationIndex:
    def test_builds_index_from_multiple_batches(self) -> None:
        results = _make_batch_results()
        index = SynthesisPipeline._build_citation_index(results)

        # Pages 1 and 5 should be in the index (confidence >= 0.5)
        assert 1 in index
        assert 5 in index

    def test_filters_low_confidence(self) -> None:
        results = _make_batch_results()
        index = SynthesisPipeline._build_citation_index(results)

        # Page 10 is from q4 with confidence 0.2 — should be excluded
        assert 10 not in index

    def test_deduplicates_by_page_and_quote_prefix(self) -> None:
        results = [
            BatchExtractionResult(
                category=ExtractionCategory.DEMOGRAPHICS,
                retrieval=RetrievalResult(query="demographics"),
                extractions=[
                    ExtractionResult(
                        question_id="q1",
                        answer="John Doe",
                        confidence=0.95,
                        citations=[
                            Citation(page_number=1, verbatim_quote="Patient: John Doe"),
                        ],
                    ),
                    ExtractionResult(
                        question_id="q2",
                        answer="John Doe",
                        confidence=0.85,
                        citations=[
                            Citation(page_number=1, verbatim_quote="Patient: John Doe"),
                        ],
                    ),
                ],
            ),
        ]
        index = SynthesisPipeline._build_citation_index(results)

        # Same page + same quote prefix → deduplicated to 1 entry
        assert len(index[1]) == 1

    def test_empty_citations(self) -> None:
        results = [
            BatchExtractionResult(
                category=ExtractionCategory.DEMOGRAPHICS,
                retrieval=RetrievalResult(query="demographics"),
                extractions=[
                    ExtractionResult(
                        question_id="q1",
                        answer="unknown",
                        confidence=0.9,
                        citations=[],
                    ),
                ],
            ),
        ]
        index = SynthesisPipeline._build_citation_index(results)
        assert index == {}

    def test_empty_results(self) -> None:
        index = SynthesisPipeline._build_citation_index([])
        assert index == {}

    def test_citation_ref_has_source_info(self) -> None:
        results = _make_batch_results()
        index = SynthesisPipeline._build_citation_index(results)

        page_1_refs = index[1]
        assert len(page_1_refs) >= 1
        ref = page_1_refs[0]
        assert ref.page_number == 1
        assert ref.source_type == "face_sheet"
        assert ref.section_title == "Face Sheet"


class TestCategoryPropagation:
    def test_citations_include_full_objects(self) -> None:
        """_prepare_category_summaries now includes full citation dicts."""
        client = LLMClient(_make_settings())
        pipeline = SynthesisPipeline(client)
        results = _make_batch_results()

        summaries = pipeline._prepare_category_summaries(results)

        demo_answers = summaries[0]["answers"]
        assert len(demo_answers) == 2

        # Citations should be dicts with page_number, not just strings
        first_citations = demo_answers[0]["citations"]
        assert len(first_citations) == 1
        assert isinstance(first_citations[0], dict)
        assert "page_number" in first_citations[0]
        assert "section_title" in first_citations[0]
        assert "section_type" in first_citations[0]
        assert "verbatim_quote" in first_citations[0]
        assert first_citations[0]["page_number"] == 1

    def test_low_confidence_still_filtered(self) -> None:
        """Low-confidence extractions are still excluded from summaries."""
        client = LLMClient(_make_settings())
        pipeline = SynthesisPipeline(client)
        results = _make_batch_results()

        summaries = pipeline._prepare_category_summaries(results)
        lab_answers = summaries[1]["answers"]
        assert len(lab_answers) == 1  # Only q3 (0.85 confidence)

    def test_citations_limited_to_three(self) -> None:
        """At most 3 citations per answer."""
        client = LLMClient(_make_settings())
        pipeline = SynthesisPipeline(client)

        results = [
            BatchExtractionResult(
                category=ExtractionCategory.DEMOGRAPHICS,
                retrieval=RetrievalResult(query="demographics"),
                extractions=[
                    ExtractionResult(
                        question_id="q1",
                        answer="test",
                        confidence=0.9,
                        citations=[
                            Citation(page_number=i, verbatim_quote=f"Quote {i}")
                            for i in range(5)
                        ],
                    ),
                ],
            ),
        ]

        summaries = pipeline._prepare_category_summaries(results)
        assert len(summaries[0]["answers"][0]["citations"]) == 3
