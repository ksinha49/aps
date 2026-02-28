"""Tests for the PDFFormatter."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from pageindex_rag.core.config import PDFFormattingConfig
from pageindex_rag.synthesis.models import SynthesisSection, UnderwriterSummary

# Skip the entire module if reportlab is not installed
reportlab = pytest.importorskip("reportlab")

from pageindex_rag.formatters.pdf_formatter import PDFFormatter  # noqa: E402


def _make_summary(
    sections: list[SynthesisSection] | None = None,
    risk_factors: list[str] | None = None,
) -> UnderwriterSummary:
    return UnderwriterSummary(
        document_id="test-doc",
        patient_demographics="John Doe, DOB 01/15/1960, Male",
        sections=sections
        or [
            SynthesisSection(
                title="Demographics",
                content="Patient is a 65-year-old male.",
                source_categories=["demographics"],
                key_findings=["Age 65", "Male"],
            ),
            SynthesisSection(
                title="Lab Results",
                content="WBC 7.2 normal range.",
                source_categories=["lab_results"],
                key_findings=["WBC 7.2 - normal", "Minor elevation in LDL"],
            ),
        ],
        risk_factors=risk_factors or ["Age over 60", "Elevated BMI"],
        overall_assessment="Standard risk profile with moderate concerns.",
        total_questions_answered=50,
        high_confidence_count=42,
        generated_at="2026-02-27T00:00:00Z",
    )


class TestPDFFormatterBasic:
    def test_format_returns_bytes(self) -> None:
        result = PDFFormatter().format(_make_summary())
        assert isinstance(result, bytes)

    def test_format_produces_valid_pdf(self) -> None:
        result = PDFFormatter().format(_make_summary())
        assert result[:5] == b"%PDF-"

    def test_format_non_trivial_size(self) -> None:
        result = PDFFormatter().format(_make_summary())
        assert len(result) > 1000

    def test_content_type(self) -> None:
        assert PDFFormatter().content_type == "application/pdf"

    def test_format_to_file(self, tmp_path) -> None:
        path = tmp_path / "output.pdf"
        result = PDFFormatter().format_to_file(_make_summary(), path)
        assert result == path
        assert path.exists()
        assert path.read_bytes()[:5] == b"%PDF-"


class TestPDFFormatterCoverPage:
    def test_cover_page_enabled_by_default(self) -> None:
        fmt = PDFFormatter()
        assert fmt._config.include_cover_page is True

    def test_no_cover_page(self) -> None:
        config = PDFFormattingConfig(include_cover_page=False)
        result = PDFFormatter(config).format(_make_summary())
        # Still a valid PDF, just shorter
        assert result[:5] == b"%PDF-"

    def test_with_company_name(self) -> None:
        config = PDFFormattingConfig(company_name="Acme Insurance Co")
        result = PDFFormatter(config).format(_make_summary())
        assert len(result) > 1000


class TestPDFFormatterSections:
    def test_empty_sections(self) -> None:
        summary = _make_summary(sections=[])
        result = PDFFormatter().format(summary)
        assert result[:5] == b"%PDF-"

    def test_multiple_sections(self) -> None:
        sections = [
            SynthesisSection(title=f"Section {i}", content=f"Content {i}")
            for i in range(5)
        ]
        summary = _make_summary(sections=sections)
        result = PDFFormatter().format(summary)
        assert len(result) > 1000

    def test_section_with_source_categories(self) -> None:
        sections = [
            SynthesisSection(
                title="Lab Results",
                content="All normal.",
                source_categories=["lab_results", "vital_signs"],
                key_findings=["Normal"],
            )
        ]
        summary = _make_summary(sections=sections)
        result = PDFFormatter().format(summary)
        assert result[:5] == b"%PDF-"


class TestPDFFormatterRiskFactors:
    def test_no_risk_factors(self) -> None:
        summary = _make_summary(risk_factors=[])
        result = PDFFormatter().format(summary)
        assert result[:5] == b"%PDF-"

    def test_many_risk_factors(self) -> None:
        factors = [f"Risk factor {i}" for i in range(10)]
        summary = _make_summary(risk_factors=factors)
        result = PDFFormatter().format(summary)
        assert len(result) > 1000


class TestPDFFormatterSeverityDetection:
    def test_critical_keywords(self) -> None:
        assert PDFFormatter._detect_severity("Critical cardiac event") == "CRITICAL"
        assert PDFFormatter._detect_severity("Severe anemia") == "CRITICAL"

    def test_significant_keywords(self) -> None:
        assert PDFFormatter._detect_severity("Significant weight loss") == "SIGNIFICANT"
        assert PDFFormatter._detect_severity("Major surgery") == "SIGNIFICANT"

    def test_moderate_keywords(self) -> None:
        assert PDFFormatter._detect_severity("Moderate hypertension") == "MODERATE"

    def test_minor_keywords(self) -> None:
        assert PDFFormatter._detect_severity("Minor bruising noted") == "MINOR"
        assert PDFFormatter._detect_severity("Mild discomfort") == "MINOR"

    def test_informational_fallback(self) -> None:
        assert PDFFormatter._detect_severity("WBC 7.2 in range") == "INFORMATIONAL"


class TestPDFFormatterAppendix:
    def test_appendix_disabled(self) -> None:
        config = PDFFormattingConfig(include_appendix=False)
        result = PDFFormatter(config).format(_make_summary())
        assert result[:5] == b"%PDF-"

    def test_appendix_with_batch_results(self) -> None:
        @dataclass
        class FakeExtraction:
            question_id: str = "Q1"
            answer: str = "Yes"
            confidence: float = 0.95
            source_pages: list[int] = field(default_factory=lambda: [1, 2])

        @dataclass
        class FakeBatch:
            extractions: list[FakeExtraction] = field(default_factory=lambda: [FakeExtraction()])

        result = PDFFormatter().format(_make_summary(), batch_results=[FakeBatch()])
        assert result[:5] == b"%PDF-"
        assert len(result) > 2000

    def test_appendix_with_empty_batch(self) -> None:
        @dataclass
        class FakeBatch:
            extractions: list = field(default_factory=list)

        result = PDFFormatter().format(_make_summary(), batch_results=[FakeBatch()])
        assert result[:5] == b"%PDF-"


class TestPDFFormatterPageSize:
    def test_a4_page_size(self) -> None:
        config = PDFFormattingConfig(page_size="a4")
        result = PDFFormatter(config).format(_make_summary())
        assert result[:5] == b"%PDF-"

    def test_letter_page_size(self) -> None:
        config = PDFFormattingConfig(page_size="letter")
        result = PDFFormatter(config).format(_make_summary())
        assert result[:5] == b"%PDF-"
