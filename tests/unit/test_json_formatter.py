"""Tests for the JSONFormatter."""

from __future__ import annotations

import json

from pageindex_rag.formatters.json_formatter import JSONFormatter
from pageindex_rag.synthesis.models import SynthesisSection, UnderwriterSummary


def _make_summary() -> UnderwriterSummary:
    return UnderwriterSummary(
        document_id="doc-001",
        patient_demographics="Jane Doe, DOB 03/22/1975, Female",
        sections=[
            SynthesisSection(
                title="Demographics",
                content="Patient is a 50-year-old female.",
                source_categories=["demographics"],
                key_findings=["Age 50", "Female"],
            ),
        ],
        risk_factors=["Elevated BMI"],
        overall_assessment="Standard risk.",
        total_questions_answered=30,
        high_confidence_count=25,
        generated_at="2026-02-27T00:00:00Z",
    )


class TestJSONFormatter:
    def test_format_returns_bytes(self) -> None:
        result = JSONFormatter().format(_make_summary())
        assert isinstance(result, bytes)

    def test_format_is_valid_json(self) -> None:
        result = JSONFormatter().format(_make_summary())
        parsed = json.loads(result)
        assert parsed["document_id"] == "doc-001"

    def test_round_trip_fields(self) -> None:
        summary = _make_summary()
        result = json.loads(JSONFormatter().format(summary))
        assert result["patient_demographics"] == summary.patient_demographics
        assert result["total_questions_answered"] == 30
        assert result["high_confidence_count"] == 25
        assert len(result["sections"]) == 1
        assert result["sections"][0]["title"] == "Demographics"
        assert result["risk_factors"] == ["Elevated BMI"]

    def test_content_type(self) -> None:
        assert JSONFormatter().content_type == "application/json"

    def test_format_to_file(self, tmp_path) -> None:
        path = tmp_path / "out.json"
        result = JSONFormatter().format_to_file(_make_summary(), path)
        assert result == path
        assert path.exists()
        parsed = json.loads(path.read_text())
        assert parsed["document_id"] == "doc-001"
