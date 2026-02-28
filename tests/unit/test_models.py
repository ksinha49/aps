"""Unit tests for data models."""

from datetime import datetime, timezone

import pytest

from pageindex_rag.models import (
    BatchExtractionResult,
    Citation,
    DocumentIndex,
    ExtractionCategory,
    ExtractionQuestion,
    ExtractionResult,
    MedicalSectionType,
    PageContent,
    RetrievalResult,
    TreeNode,
)


class TestPageContent:
    def test_create_minimal(self):
        p = PageContent(page_number=1, text="hello")
        assert p.page_number == 1
        assert p.text == "hello"
        assert p.token_count is None

    def test_create_with_tokens(self):
        p = PageContent(page_number=5, text="world", token_count=42)
        assert p.token_count == 42

    def test_serialization_roundtrip(self):
        p = PageContent(page_number=3, text="test data", token_count=10)
        d = p.model_dump()
        restored = PageContent(**d)
        assert restored == p


class TestTreeNode:
    def test_leaf_node(self):
        node = TreeNode(title="Section A", start_index=1, end_index=5)
        assert node.children == []
        assert node.node_id == ""
        assert node.content_type == MedicalSectionType.UNKNOWN

    def test_nested_tree(self):
        child = TreeNode(title="Child", start_index=2, end_index=3)
        parent = TreeNode(title="Parent", start_index=1, end_index=5, children=[child])
        assert len(parent.children) == 1
        assert parent.children[0].title == "Child"

    def test_with_medical_type(self):
        node = TreeNode(
            title="Labs",
            start_index=1,
            end_index=1,
            content_type=MedicalSectionType.LAB_REPORT,
        )
        assert node.content_type == MedicalSectionType.LAB_REPORT


class TestDocumentIndex:
    def test_create(self, sample_tree):
        idx = DocumentIndex(
            doc_id="test-001",
            doc_name="Test APS",
            total_pages=10,
            tree=sample_tree,
        )
        assert idx.doc_id == "test-001"
        assert idx.total_pages == 10
        assert len(idx.tree) == 1
        assert isinstance(idx.created_at, datetime)

    def test_json_roundtrip(self, sample_tree):
        idx = DocumentIndex(
            doc_id="test-002",
            doc_name="Roundtrip",
            total_pages=5,
            tree=sample_tree,
        )
        json_str = idx.model_dump_json()
        restored = DocumentIndex.model_validate_json(json_str)
        assert restored.doc_id == idx.doc_id
        assert len(restored.tree) == len(idx.tree)


class TestCitation:
    def test_create_minimal(self):
        c = Citation(page_number=5, verbatim_quote="WBC 7.2")
        assert c.page_number == 5
        assert c.section_title == ""
        assert c.section_type == ""
        assert c.verbatim_quote == "WBC 7.2"

    def test_create_full(self):
        c = Citation(
            page_number=5,
            section_title="Laboratory Report",
            section_type="lab_report",
            verbatim_quote="HbA1c: 7.2% (< 5.7) HIGH",
        )
        assert c.section_title == "Laboratory Report"
        assert c.section_type == "lab_report"

    def test_serialization_roundtrip(self):
        c = Citation(
            page_number=10,
            section_title="Discharge Summary",
            section_type="discharge_summary",
            verbatim_quote="Condition at discharge: Stable",
        )
        d = c.model_dump()
        restored = Citation(**d)
        assert restored == c

    def test_json_roundtrip(self):
        c = Citation(page_number=3, verbatim_quote="BP 130/85")
        json_str = c.model_dump_json()
        restored = Citation.model_validate_json(json_str)
        assert restored == c


class TestExtractionModels:
    def test_question(self):
        q = ExtractionQuestion(
            question_id="q001",
            category=ExtractionCategory.DEMOGRAPHICS,
            question_text="What is the patient's name?",
        )
        assert q.tier == 1
        assert q.expected_type == "text"

    def test_retrieval_result(self):
        r = RetrievalResult(query="blood pressure")
        assert r.retrieved_nodes == []
        assert r.source_pages == []

    def test_extraction_result_confidence_bounds(self):
        r = ExtractionResult(question_id="q001", confidence=0.95)
        assert r.confidence == 0.95

        with pytest.raises(Exception):
            ExtractionResult(question_id="q001", confidence=1.5)

    def test_extraction_result_with_citations(self):
        cit = Citation(
            page_number=5,
            section_title="Lab Report",
            section_type="lab_report",
            verbatim_quote="WBC 7.2",
        )
        r = ExtractionResult(
            question_id="q001",
            answer="7.2",
            confidence=0.95,
            citations=[cit],
            source_pages=[5],
            evidence_text="WBC 7.2",
        )
        assert len(r.citations) == 1
        assert r.citations[0].page_number == 5
        assert r.source_pages == [5]

    def test_extraction_result_backward_compat_no_citations(self):
        """ExtractionResult works without citations (backward compat)."""
        r = ExtractionResult(
            question_id="q001",
            answer="test",
            source_pages=[1, 2],
            evidence_text="some evidence",
        )
        assert r.citations == []
        assert r.source_pages == [1, 2]
        assert r.evidence_text == "some evidence"

    def test_extraction_result_json_roundtrip_with_citations(self):
        cit = Citation(page_number=7, verbatim_quote="Ibuprofen 800mg TID")
        r = ExtractionResult(
            question_id="med-001",
            answer="Ibuprofen 800mg TID",
            confidence=0.9,
            citations=[cit],
            source_pages=[7],
            evidence_text="Ibuprofen 800mg TID",
        )
        json_str = r.model_dump_json()
        restored = ExtractionResult.model_validate_json(json_str)
        assert len(restored.citations) == 1
        assert restored.citations[0].page_number == 7

    def test_batch_result(self):
        br = BatchExtractionResult(
            category=ExtractionCategory.LAB_RESULTS,
            retrieval=RetrievalResult(query="lab values"),
        )
        assert br.extractions == []


class TestEnums:
    def test_medical_section_values(self):
        assert MedicalSectionType.FACE_SHEET.value == "face_sheet"
        assert len(MedicalSectionType) == 15

    def test_extraction_category_values(self):
        assert ExtractionCategory.DEMOGRAPHICS.value == "demographics"
        assert len(ExtractionCategory) == 16
