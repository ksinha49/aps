"""Tests that formatter implementations satisfy the IOutputFormatter protocol."""

from __future__ import annotations

import pytest

from pageindex_rag.formatters.json_formatter import JSONFormatter
from pageindex_rag.formatters.protocols import IOutputFormatter


class TestJSONFormatterSatisfiesProtocol:
    def test_isinstance_check(self) -> None:
        assert isinstance(JSONFormatter(), IOutputFormatter)

    def test_has_format_method(self) -> None:
        assert callable(getattr(JSONFormatter, "format", None))

    def test_has_format_to_file_method(self) -> None:
        assert callable(getattr(JSONFormatter, "format_to_file", None))

    def test_has_content_type_property(self) -> None:
        assert hasattr(JSONFormatter, "content_type")


class TestPDFFormatterSatisfiesProtocol:
    @pytest.fixture(autouse=True)
    def _skip_if_no_reportlab(self) -> None:
        pytest.importorskip("reportlab")

    def test_isinstance_check(self) -> None:
        from pageindex_rag.formatters.pdf_formatter import PDFFormatter

        assert isinstance(PDFFormatter(), IOutputFormatter)

    def test_has_format_method(self) -> None:
        from pageindex_rag.formatters.pdf_formatter import PDFFormatter

        assert callable(getattr(PDFFormatter, "format", None))

    def test_has_format_to_file_method(self) -> None:
        from pageindex_rag.formatters.pdf_formatter import PDFFormatter

        assert callable(getattr(PDFFormatter, "format_to_file", None))

    def test_has_content_type_property(self) -> None:
        from pageindex_rag.formatters.pdf_formatter import PDFFormatter

        assert hasattr(PDFFormatter, "content_type")
