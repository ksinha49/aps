"""Tests that formatter implementations satisfy the IOutputFormatter protocol."""

from __future__ import annotations

import pytest

from scout_ai.formatters.json_formatter import JSONFormatter
from scout_ai.formatters.protocols import IOutputFormatter
from scout_ai.synthesis.models import APSSummary, PatientDemographics, UnderwriterSummary


class TestJSONFormatterSatisfiesProtocol:
    def test_isinstance_check(self) -> None:
        assert isinstance(JSONFormatter(), IOutputFormatter)

    def test_has_format_method(self) -> None:
        assert callable(getattr(JSONFormatter, "format", None))

    def test_has_format_to_file_method(self) -> None:
        assert callable(getattr(JSONFormatter, "format_to_file", None))

    def test_has_content_type_property(self) -> None:
        assert hasattr(JSONFormatter, "content_type")

    def test_accepts_aps_summary(self) -> None:
        """JSONFormatter can serialize an APSSummary."""
        summary = APSSummary(
            document_id="test",
            demographics=PatientDemographics(full_name="Jane Doe"),
        )
        result = JSONFormatter().format(summary)
        assert b"test" in result
        assert b"Jane Doe" in result


class TestPDFFormatterSatisfiesProtocol:
    @pytest.fixture(autouse=True)
    def _skip_if_no_reportlab(self) -> None:
        pytest.importorskip("reportlab")

    def test_isinstance_check(self) -> None:
        from scout_ai.formatters.pdf_formatter import PDFFormatter

        assert isinstance(PDFFormatter(), IOutputFormatter)

    def test_has_format_method(self) -> None:
        from scout_ai.formatters.pdf_formatter import PDFFormatter

        assert callable(getattr(PDFFormatter, "format", None))

    def test_has_format_to_file_method(self) -> None:
        from scout_ai.formatters.pdf_formatter import PDFFormatter

        assert callable(getattr(PDFFormatter, "format_to_file", None))

    def test_has_content_type_property(self) -> None:
        from scout_ai.formatters.pdf_formatter import PDFFormatter

        assert hasattr(PDFFormatter, "content_type")

    def test_accepts_aps_summary(self) -> None:
        from scout_ai.formatters.pdf_formatter import PDFFormatter

        summary = APSSummary(
            document_id="test",
            demographics=PatientDemographics(full_name="Jane Doe"),
        )
        result = PDFFormatter().format(summary)
        assert result[:5] == b"%PDF-"


class TestSummaryInputType:
    def test_union_includes_underwriter_summary(self) -> None:
        summary = UnderwriterSummary(document_id="test", patient_demographics="Test")
        assert isinstance(summary, (UnderwriterSummary, APSSummary))

    def test_union_includes_aps_summary(self) -> None:
        summary = APSSummary(document_id="test")
        assert isinstance(summary, (UnderwriterSummary, APSSummary))
