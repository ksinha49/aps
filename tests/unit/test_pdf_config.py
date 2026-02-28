"""Tests for PDFFormattingConfig defaults and env overrides."""

from __future__ import annotations

from pageindex_rag.core.config import PDFFormattingConfig


class TestPDFFormattingConfigDefaults:
    def test_page_size_default(self) -> None:
        cfg = PDFFormattingConfig()
        assert cfg.page_size == "letter"

    def test_margin_default(self) -> None:
        cfg = PDFFormattingConfig()
        assert cfg.margin_inches == 0.75

    def test_font_family_default(self) -> None:
        cfg = PDFFormattingConfig()
        assert cfg.font_family == "Helvetica"

    def test_body_font_size_default(self) -> None:
        cfg = PDFFormattingConfig()
        assert cfg.body_font_size == 10

    def test_heading_font_size_default(self) -> None:
        cfg = PDFFormattingConfig()
        assert cfg.heading_font_size == 14

    def test_include_appendix_default(self) -> None:
        cfg = PDFFormattingConfig()
        assert cfg.include_appendix is True

    def test_include_cover_page_default(self) -> None:
        cfg = PDFFormattingConfig()
        assert cfg.include_cover_page is True

    def test_company_name_default(self) -> None:
        cfg = PDFFormattingConfig()
        assert cfg.company_name == ""

    def test_confidential_watermark_default(self) -> None:
        cfg = PDFFormattingConfig()
        assert cfg.confidential_watermark is True


class TestPDFFormattingConfigEnvOverrides:
    def test_page_size_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("PAGEINDEX_PDF_PAGE_SIZE", "a4")
        cfg = PDFFormattingConfig()
        assert cfg.page_size == "a4"

    def test_margin_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("PAGEINDEX_PDF_MARGIN_INCHES", "1.0")
        cfg = PDFFormattingConfig()
        assert cfg.margin_inches == 1.0

    def test_include_appendix_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("PAGEINDEX_PDF_INCLUDE_APPENDIX", "false")
        cfg = PDFFormattingConfig()
        assert cfg.include_appendix is False

    def test_company_name_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("PAGEINDEX_PDF_COMPANY_NAME", "Acme Insurance")
        cfg = PDFFormattingConfig()
        assert cfg.company_name == "Acme Insurance"

    def test_confidential_watermark_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("PAGEINDEX_PDF_CONFIDENTIAL_WATERMARK", "false")
        cfg = PDFFormattingConfig()
        assert cfg.confidential_watermark is False


class TestPDFConfigInAppSettings:
    def test_app_settings_has_pdf(self) -> None:
        from pageindex_rag.core.config import AppSettings

        settings = AppSettings()
        assert hasattr(settings, "pdf")
        assert isinstance(settings.pdf, PDFFormattingConfig)
