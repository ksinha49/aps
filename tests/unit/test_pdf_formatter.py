"""Tests for the PDFFormatter."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from scout_ai.core.config import PDFFormattingConfig
from scout_ai.synthesis.models import (
    Allergy,
    APSSection,
    APSSummary,
    CitationRef,
    Condition,
    Encounter,
    Finding,
    LabResult,
    Medication,
    PatientDemographics,
    RedFlag,
    RiskClassification,
    SynthesisSection,
    UnderwriterSummary,
    VitalSign,
)

# Skip the entire module if reportlab is not installed
reportlab = pytest.importorskip("reportlab")

from scout_ai.formatters.pdf_formatter import PDFFormatter  # noqa: E402


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


def _make_aps_summary() -> APSSummary:
    return APSSummary(
        document_id="test-aps-doc",
        demographics=PatientDemographics(
            full_name="John Doe",
            date_of_birth="01/15/1960",
            age="65",
            gender="Male",
            insurance_id="INS-12345",
        ),
        sections=[
            APSSection(
                section_key="medical_history",
                section_number="3.0",
                title="Medical History",
                content="Patient has history of hypertension and type 2 diabetes.",
                source_categories=["medical_history", "diagnoses"],
                findings=[
                    Finding(
                        text="Hypertension diagnosed 2015",
                        severity="MODERATE",
                        citations=[
                            CitationRef(page_number=12, date="03/2024", source_type="Progress Note"),
                        ],
                    ),
                    Finding(
                        text="Type 2 Diabetes, HbA1c 7.2%",
                        severity="SIGNIFICANT",
                        citations=[
                            CitationRef(page_number=15, date="01/2024", source_type="Lab Report"),
                        ],
                    ),
                ],
                conditions=[
                    Condition(name="Hypertension", icd10_code="I10", onset_date="2015", status="active"),
                    Condition(name="Type 2 Diabetes", icd10_code="E11.9", onset_date="2018", status="active"),
                ],
            ),
            APSSection(
                section_key="lab_results",
                section_number="10.0",
                title="Laboratory Results",
                content="Recent labs show controlled diabetes.",
                lab_results=[
                    LabResult(
                        test_name="HbA1c", value="7.2", unit="%",
                        reference_range="4.0-5.6", flag="H", date="01/2024",
                    ),
                    LabResult(
                        test_name="Glucose", value="142", unit="mg/dL",
                        reference_range="70-100", flag="H", date="01/2024",
                    ),
                    LabResult(
                        test_name="WBC", value="7.2", unit="K/uL",
                        reference_range="4.5-11.0", flag="", date="01/2024",
                    ),
                ],
            ),
            APSSection(
                section_key="medications",
                title="Medications",
                medications=[
                    Medication(name="Metformin", dose="500mg", frequency="BID", route="oral", prescriber="Dr. Smith"),
                    Medication(name="Lisinopril", dose="10mg", frequency="QD", route="oral", prescriber="Dr. Smith"),
                ],
            ),
        ],
        risk_classification=RiskClassification(
            tier="Standard",
            table_rating="Table 2",
            rationale="Controlled comorbidities with regular monitoring.",
        ),
        risk_factors=["Age over 60", "Elevated HbA1c"],
        red_flags=[
            RedFlag(
                description="Two concurrent controlled substances",
                severity="SIGNIFICANT",
                category="medication",
            ),
        ],
        overall_assessment="Standard risk with table 2 rating due to controlled comorbidities.",
        citation_index={
            12: [CitationRef(page_number=12, source_type="progress_note", section_title="Visit Note")],
            15: [CitationRef(page_number=15, source_type="lab_report", section_title="Chemistry Panel")],
        },
        total_questions_answered=50,
        high_confidence_count=42,
        generated_at="2026-02-28T00:00:00Z",
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


# ── APS Summary tests ───────────────────────────────────────────────


class TestPDFFormatterAPSSummary:
    def test_format_returns_valid_pdf(self) -> None:
        result = PDFFormatter().format(_make_aps_summary())
        assert result[:5] == b"%PDF-"

    def test_format_non_trivial_size(self) -> None:
        result = PDFFormatter().format(_make_aps_summary())
        assert len(result) > 2000

    def test_demographics_grid(self) -> None:
        """APS with full demographics renders without error."""
        result = PDFFormatter().format(_make_aps_summary())
        assert result[:5] == b"%PDF-"

    def test_demographics_raw_text_fallback(self) -> None:
        summary = APSSummary(
            document_id="doc-1",
            demographics=PatientDemographics(raw_text="John Doe, 65, Male"),
        )
        result = PDFFormatter().format(summary)
        assert result[:5] == b"%PDF-"

    def test_lab_table(self) -> None:
        """Sections with lab_results produce a valid PDF."""
        result = PDFFormatter().format(_make_aps_summary())
        assert result[:5] == b"%PDF-"
        assert len(result) > 2000

    def test_medication_table(self) -> None:
        """Sections with medications produce a valid PDF."""
        result = PDFFormatter().format(_make_aps_summary())
        assert result[:5] == b"%PDF-"

    def test_risk_badge(self) -> None:
        """Risk classification badge renders without error."""
        result = PDFFormatter().format(_make_aps_summary())
        assert result[:5] == b"%PDF-"

    def test_risk_badge_disabled(self) -> None:
        config = PDFFormattingConfig(risk_badge_enabled=False)
        result = PDFFormatter(config).format(_make_aps_summary())
        assert result[:5] == b"%PDF-"

    def test_red_flags(self) -> None:
        """Red flags section renders without error."""
        result = PDFFormatter().format(_make_aps_summary())
        assert result[:5] == b"%PDF-"

    def test_red_flags_disabled(self) -> None:
        config = PDFFormattingConfig(red_flag_alerts=False)
        result = PDFFormatter(config).format(_make_aps_summary())
        assert result[:5] == b"%PDF-"

    def test_toc_included(self) -> None:
        """TOC is generated by default."""
        config = PDFFormattingConfig(include_toc=True)
        result = PDFFormatter(config).format(_make_aps_summary())
        assert result[:5] == b"%PDF-"

    def test_toc_disabled(self) -> None:
        config = PDFFormattingConfig(include_toc=False)
        result = PDFFormatter(config).format(_make_aps_summary())
        assert result[:5] == b"%PDF-"

    def test_citation_refs(self) -> None:
        """Citations in findings render without error."""
        config = PDFFormattingConfig(include_citation_refs=True)
        result = PDFFormatter(config).format(_make_aps_summary())
        assert result[:5] == b"%PDF-"

    def test_citation_refs_disabled(self) -> None:
        config = PDFFormattingConfig(include_citation_refs=False)
        result = PDFFormatter(config).format(_make_aps_summary())
        assert result[:5] == b"%PDF-"

    def test_section_numbering(self) -> None:
        config = PDFFormattingConfig(section_numbering=True)
        result = PDFFormatter(config).format(_make_aps_summary())
        assert result[:5] == b"%PDF-"

    def test_section_numbering_disabled(self) -> None:
        config = PDFFormattingConfig(section_numbering=False)
        result = PDFFormatter(config).format(_make_aps_summary())
        assert result[:5] == b"%PDF-"

    def test_backward_compat_legacy_still_works(self) -> None:
        """Passing UnderwriterSummary still uses legacy path."""
        legacy = _make_summary()
        result = PDFFormatter().format(legacy)
        assert result[:5] == b"%PDF-"

    def test_empty_aps_summary(self) -> None:
        summary = APSSummary(document_id="empty")
        result = PDFFormatter().format(summary)
        assert result[:5] == b"%PDF-"

    def test_aps_with_encounters(self) -> None:
        summary = APSSummary(
            document_id="enc-doc",
            sections=[
                APSSection(
                    section_key="encounter_chronology",
                    title="Encounters",
                    encounters=[
                        Encounter(
                            date="2024-01-15",
                            provider="Dr. Smith",
                            encounter_type="office visit",
                            summary="Routine checkup",
                        ),
                        Encounter(
                            date="2024-03-20",
                            provider="Dr. Jones",
                            encounter_type="telehealth",
                            summary="Follow-up",
                        ),
                    ],
                ),
            ],
        )
        result = PDFFormatter().format(summary)
        assert result[:5] == b"%PDF-"

    def test_aps_with_vital_signs(self) -> None:
        summary = APSSummary(
            document_id="vitals-doc",
            sections=[
                APSSection(
                    section_key="build_and_vitals",
                    title="Vitals",
                    vital_signs=[
                        VitalSign(name="BP", value="120/80 mmHg", date="2024-03-15"),
                        VitalSign(name="HR", value="72 bpm", date="2024-03-15"),
                    ],
                ),
            ],
        )
        result = PDFFormatter().format(summary)
        assert result[:5] == b"%PDF-"

    def test_aps_with_allergies(self) -> None:
        summary = APSSummary(
            document_id="allergy-doc",
            sections=[
                APSSection(
                    section_key="allergies",
                    title="Allergies",
                    allergies=[
                        Allergy(allergen="Penicillin", reaction="Rash", severity="moderate"),
                    ],
                ),
            ],
        )
        result = PDFFormatter().format(summary)
        assert result[:5] == b"%PDF-"

    def test_format_to_file_aps(self, tmp_path) -> None:
        path = tmp_path / "aps_output.pdf"
        result = PDFFormatter().format_to_file(_make_aps_summary(), path)
        assert result == path
        assert path.exists()
        assert path.read_bytes()[:5] == b"%PDF-"
