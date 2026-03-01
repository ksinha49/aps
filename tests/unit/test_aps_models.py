"""Tests for APS Schema v1.0.0 data models."""

from __future__ import annotations

from scout_ai.synthesis.models import (
    Allergy,
    APSSection,
    APSSummary,
    CitationRef,
    Condition,
    Encounter,
    Finding,
    ImagingResult,
    LabResult,
    Medication,
    PatientDemographics,
    RedFlag,
    RiskClassification,
    SurgicalHistory,
    UnderwriterSummary,
    VitalSign,
)


class TestCitationRef:
    def test_display_page_only(self) -> None:
        ref = CitationRef(page_number=42)
        assert ref.display() == "p.42"

    def test_display_with_date(self) -> None:
        ref = CitationRef(page_number=42, date="03/2024")
        assert "p.42" in ref.display()
        assert "03/2024" in ref.display()

    def test_display_with_source_type(self) -> None:
        ref = CitationRef(page_number=42, date="03/2024", source_type="Progress Note")
        result = ref.display()
        assert "p.42" in result
        assert "03/2024" in result
        assert "Progress Note" in result

    def test_display_full(self) -> None:
        ref = CitationRef(
            page_number=42, date="03/2024", source_type="Lab Report",
            section_title="Chemistry Panel", verbatim_quote="HbA1c 7.2%",
        )
        result = ref.display()
        assert "p.42" in result
        assert "03/2024" in result
        assert "Lab Report" in result

    def test_defaults(self) -> None:
        ref = CitationRef(page_number=1)
        assert ref.date == ""
        assert ref.source_type == ""
        assert ref.section_title == ""
        assert ref.verbatim_quote == ""


class TestFinding:
    def test_defaults(self) -> None:
        f = Finding(text="Hypertension diagnosed")
        assert f.severity == "INFORMATIONAL"
        assert f.citations == []

    def test_severity(self) -> None:
        f = Finding(text="Critical arrhythmia", severity="CRITICAL")
        assert f.severity == "CRITICAL"

    def test_with_citations(self) -> None:
        f = Finding(
            text="Elevated BP",
            severity="MODERATE",
            citations=[CitationRef(page_number=5)],
        )
        assert len(f.citations) == 1
        assert f.citations[0].page_number == 5


class TestPatientDemographics:
    def test_defaults(self) -> None:
        d = PatientDemographics()
        assert d.full_name == ""
        assert d.date_of_birth == ""
        assert d.raw_text == ""

    def test_summary_text_with_full_name(self) -> None:
        d = PatientDemographics(
            full_name="John Doe", date_of_birth="01/15/1960", age="65", gender="Male"
        )
        result = d.summary_text()
        assert "John Doe" in result
        assert "DOB 01/15/1960" in result
        assert "Age 65" in result
        assert "Male" in result

    def test_summary_text_fallback_to_raw(self) -> None:
        d = PatientDemographics(raw_text="Patient is a 65-year-old male")
        assert d.summary_text() == "Patient is a 65-year-old male"

    def test_summary_text_no_data(self) -> None:
        d = PatientDemographics()
        assert d.summary_text() == "Demographics not available"

    def test_summary_text_partial(self) -> None:
        d = PatientDemographics(full_name="Jane Doe")
        result = d.summary_text()
        assert "Jane Doe" in result
        assert "DOB" not in result


class TestCondition:
    def test_with_icd10(self) -> None:
        c = Condition(name="Hypertension", icd10_code="I10", status="active")
        assert c.icd10_code == "I10"
        assert c.citations == []

    def test_defaults(self) -> None:
        c = Condition(name="Diabetes")
        assert c.onset_date == ""
        assert c.severity == ""


class TestMedication:
    def test_full(self) -> None:
        m = Medication(name="Metformin", dose="500mg", frequency="BID", route="oral")
        assert m.prescriber == ""
        assert m.start_date == ""


class TestLabResult:
    def test_flag_values(self) -> None:
        assert LabResult(test_name="HbA1c", flag="H").flag == "H"
        assert LabResult(test_name="WBC", flag="").flag == ""
        assert LabResult(test_name="Glucose", flag="L").flag == "L"

    def test_defaults(self) -> None:
        lr = LabResult(test_name="BMP")
        assert lr.value == ""
        assert lr.unit == ""
        assert lr.reference_range == ""
        assert lr.date == ""


class TestImagingResult:
    def test_defaults(self) -> None:
        ir = ImagingResult(modality="MRI")
        assert ir.body_part == ""
        assert ir.finding == ""
        assert ir.impression == ""


class TestEncounter:
    def test_full(self) -> None:
        e = Encounter(date="2024-03-15", provider="Dr. Smith", encounter_type="office visit")
        assert e.summary == ""


class TestVitalSign:
    def test_full(self) -> None:
        vs = VitalSign(name="BP", value="120/80 mmHg", date="2024-03-15")
        assert vs.flag == ""


class TestAllergy:
    def test_defaults(self) -> None:
        a = Allergy(allergen="Penicillin")
        assert a.reaction == ""
        assert a.severity == ""


class TestSurgicalHistory:
    def test_defaults(self) -> None:
        sh = SurgicalHistory(procedure="Appendectomy")
        assert sh.date == ""
        assert sh.outcome == ""
        assert sh.complications == ""


class TestRiskClassification:
    def test_defaults(self) -> None:
        rc = RiskClassification()
        assert rc.tier == ""
        assert rc.table_rating == ""
        assert rc.confidence == 0.0

    def test_full(self) -> None:
        rc = RiskClassification(
            tier="Standard", table_rating="Table 2",
            rationale="Controlled comorbidities", confidence=0.85,
        )
        assert rc.tier == "Standard"


class TestRedFlag:
    def test_defaults(self) -> None:
        rf = RedFlag(description="Concurrent controlled substances")
        assert rf.severity == "MODERATE"
        assert rf.category == ""
        assert rf.citations == []


class TestAPSSection:
    def test_defaults(self) -> None:
        s = APSSection(section_key="medical_history")
        assert s.title == ""
        assert s.content == ""
        assert s.findings == []
        assert s.conditions == []
        assert s.medications == []
        assert s.lab_results == []

    def test_with_typed_data(self) -> None:
        s = APSSection(
            section_key="lab_results",
            title="Lab Results",
            lab_results=[LabResult(test_name="HbA1c", value="7.2", flag="H")],
            findings=[Finding(text="Elevated HbA1c", severity="SIGNIFICANT")],
        )
        assert len(s.lab_results) == 1
        assert len(s.findings) == 1


class TestAPSSummary:
    def test_defaults(self) -> None:
        s = APSSummary(document_id="doc-1")
        assert s.demographics.full_name == ""
        assert s.sections == []
        assert s.risk_factors == []
        assert s.red_flags == []
        assert s.citation_index == {}

    def test_to_underwriter_summary(self) -> None:
        summary = APSSummary(
            document_id="test-doc",
            demographics=PatientDemographics(
                full_name="John Doe", date_of_birth="01/15/1960", age="65", gender="Male"
            ),
            sections=[
                APSSection(
                    section_key="medical_history",
                    title="Medical History",
                    content="Patient has hypertension.",
                    source_categories=["medical_history"],
                    findings=[
                        Finding(text="Hypertension since 2015", severity="MODERATE"),
                        Finding(text="Controlled with medication", severity="MINOR"),
                    ],
                ),
            ],
            risk_factors=["Age over 60"],
            overall_assessment="Standard risk.",
            total_questions_answered=50,
            high_confidence_count=42,
            generated_at="2026-02-28T00:00:00Z",
        )

        uw = summary.to_underwriter_summary()

        assert isinstance(uw, UnderwriterSummary)
        assert uw.document_id == "test-doc"
        assert "John Doe" in uw.patient_demographics
        assert "DOB 01/15/1960" in uw.patient_demographics
        assert len(uw.sections) == 1
        assert uw.sections[0].title == "Medical History"
        assert uw.sections[0].content == "Patient has hypertension."
        assert len(uw.sections[0].key_findings) == 2
        assert uw.sections[0].key_findings[0] == "Hypertension since 2015"
        assert uw.risk_factors == ["Age over 60"]
        assert uw.overall_assessment == "Standard risk."
        assert uw.total_questions_answered == 50
        assert uw.high_confidence_count == 42

    def test_to_underwriter_summary_uses_section_key_fallback(self) -> None:
        summary = APSSummary(
            document_id="doc-1",
            sections=[APSSection(section_key="allergies", content="No known allergies.")],
        )
        uw = summary.to_underwriter_summary()
        assert uw.sections[0].title == "allergies"

    def test_to_underwriter_summary_empty(self) -> None:
        summary = APSSummary(document_id="empty")
        uw = summary.to_underwriter_summary()
        assert uw.document_id == "empty"
        assert uw.sections == []
        assert uw.patient_demographics == "Demographics not available"
