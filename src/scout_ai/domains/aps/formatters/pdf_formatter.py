"""PDF output formatter using reportlab.

Renders ``UnderwriterSummary`` or ``APSSummary`` as a professionally formatted
APS summary PDF suitable for underwriter review.  Requires the ``pdf`` optional
dependency::

    pip install scout-ai[pdf]
"""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Any

from scout_ai.core.config import PDFFormattingConfig
from scout_ai.domains.aps.formatters.pdf_styles import (
    APS_SECTION_TITLES,
    ASSESSMENT_BOX_BG_COLOR,
    CATEGORY_DISPLAY_NAMES,
    CITATION_TEXT_COLOR,
    DEMOGRAPHICS_HEADER_BG,
    HEADER_BG_COLOR,
    HEADER_TEXT_COLOR,
    LAB_FLAG_COLORS,
    RED_FLAG_BG_COLOR,
    RED_FLAG_BORDER_COLOR,
    RISK_BOX_BG_COLOR,
    RISK_TIER_COLORS,
    SECTION_BORDER_COLOR,
    SEVERITY_COLORS,
    YN_COLORS,
    YN_CONDITION_DISPLAY_NAMES,
)
from scout_ai.domains.aps.models import (
    APSSection,
    APSSummary,
    CitationRef,
    Finding,
    PatientDemographics,
    RedFlag,
    RiskClassification,
    SynthesisSection,
    UnderwriterSummary,
    UnderwritingAPSSummary,
    YNCondition,
)

try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4, LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        BaseDocTemplate,
        Flowable,
        Frame,
        PageBreak,
        PageTemplate,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.platypus import (
        Paragraph as _RawParagraph,
    )
    from reportlab.platypus.tableofcontents import TableOfContents
except ImportError as _exc:
    raise ImportError(
        "reportlab is required for PDF output. Install with: pip install scout-ai[pdf]"
    ) from _exc


# ── Unicode sanitization ────────────────────────────────────────────
# Helvetica lacks glyphs for many Unicode characters that LLMs emit
# (non-breaking hyphens, narrow spaces, smart quotes, etc.).  We
# sanitize text at the Paragraph boundary so every flowable is safe.

_UNICODE_REPLACEMENTS: dict[str, str] = {
    # Dashes / hyphens
    "\u2011": "-",       # non-breaking hyphen
    "\u2010": "-",       # hyphen
    "\u2012": "-",       # figure dash
    "\u2013": "-",       # en-dash
    "\u2014": "-",       # em-dash
    "\u2015": "-",       # horizontal bar
    # Spaces
    "\u202f": " ",       # narrow no-break space
    "\u00a0": " ",       # non-breaking space
    "\u2009": " ",       # thin space
    "\u200a": " ",       # hair space
    # Quotes
    "\u2018": "'",       # left single quote
    "\u2019": "'",       # right single quote
    "\u201c": '"',       # left double quote
    "\u201d": '"',       # right double quote
    # Misc punctuation
    "\u2026": "...",     # ellipsis
    "\u00b7": ".",       # middle dot
    # Superscript digits (Helvetica lacks these glyphs)
    "\u00b2": "2",       # superscript 2
    "\u00b3": "3",       # superscript 3
    "\u2070": "0",       # superscript 0
    "\u00b9": "1",       # superscript 1
    "\u2074": "4",       # superscript 4
    "\u2075": "5",       # superscript 5
    "\u2076": "6",       # superscript 6
    "\u2077": "7",       # superscript 7
    "\u2078": "8",       # superscript 8
    "\u2079": "9",       # superscript 9
    # Scientific symbols
    "\u00b5": "u",       # micro sign → u (as in uL)
    "\u00d7": "x",       # multiplication sign → x
    "\u2192": "->",      # rightwards arrow
    "\u2191": "^",       # upwards arrow (used for "elevated")
    "\u2193": "v",       # downwards arrow (used for "decreased")
}


def _sanitize_text(text: str) -> str:
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text


def Paragraph(text: str, *args: Any, **kwargs: Any) -> _RawParagraph:  # noqa: N802
    """Sanitized Paragraph wrapper — replaces Unicode glyphs Helvetica cannot render."""
    return _RawParagraph(_sanitize_text(str(text)), *args, **kwargs)


# ── Page size lookup ─────────────────────────────────────────────────

_PAGE_SIZES = {"letter": LETTER, "a4": A4}


def _hex(color_str: str) -> HexColor:
    return HexColor(color_str)


# ── APS document template with TOC support ──────────────────────────


class _APSDocTemplate(BaseDocTemplate):
    """Document template that captures heading flowables for TOC population."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._toc_entries: list[tuple[int, str, int]] = []

    def afterFlowable(self, flowable: Flowable) -> None:
        """Register TOC entries when headings are rendered."""
        style_name = getattr(flowable, "style", None)
        if style_name is None:
            return
        name = getattr(style_name, "name", "")
        if name == "aps_heading_1":
            text = flowable.getPlainText()
            self.notify("TOCEntry", (0, text, self.page))
        elif name == "aps_heading_2":
            text = flowable.getPlainText()
            self.notify("TOCEntry", (1, text, self.page))


# ── PDFFormatter ─────────────────────────────────────────────────────


class PDFFormatter:
    """Renders ``UnderwriterSummary`` or ``APSSummary`` as a professional APS summary PDF."""

    def __init__(self, config: PDFFormattingConfig | None = None) -> None:
        self._config = config or PDFFormattingConfig()
        self._page_size: tuple[float, float] = _PAGE_SIZES.get(self._config.page_size, LETTER)
        self._margin: float = self._config.margin_inches * inch
        self._styles = self._build_styles()

    # ── Public API ───────────────────────────────────────────────────

    def format(self, summary: UnderwriterSummary, **kwargs: Any) -> bytes:
        """Render *summary* to PDF bytes.

        Dispatches to the APS-specific renderer when an ``APSSummary`` is
        passed, otherwise uses the legacy renderer.
        """
        # UnderwritingAPSSummary check must precede APSSummary (subclass)
        if isinstance(summary, UnderwritingAPSSummary):
            return self._format_underwriting(summary, **kwargs)
        if isinstance(summary, APSSummary):
            return self._format_aps(summary, **kwargs)
        return self._format_legacy(summary, **kwargs)

    def format_to_file(self, summary: UnderwriterSummary, path: Path, **kwargs: Any) -> Path:
        """Write PDF to *path* and return it."""
        path.write_bytes(self.format(summary, **kwargs))
        return path

    @property
    def content_type(self) -> str:
        return "application/pdf"

    # ── Legacy renderer (unchanged behavior) ─────────────────────────

    def _format_legacy(self, summary: UnderwriterSummary, **kwargs: Any) -> bytes:
        """Render legacy ``UnderwriterSummary`` to PDF bytes."""
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self._page_size,
            leftMargin=self._margin,
            rightMargin=self._margin,
            topMargin=self._margin + 0.3 * inch,
            bottomMargin=self._margin + 0.3 * inch,
        )
        story = self._build_story(summary, **kwargs)
        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        return buffer.getvalue()

    # ── APS renderer ─────────────────────────────────────────────────

    def _format_aps(self, summary: APSSummary, **kwargs: Any) -> bytes:
        """Render ``APSSummary`` to a professional PDF with TOC and typed tables."""
        buffer = BytesIO()
        frame = Frame(
            self._margin,
            self._margin + 0.3 * inch,
            self._content_width(),
            float(self._page_size[1]) - 2 * self._margin - 0.6 * inch,
            id="main",
        )
        template = PageTemplate(
            id="main",
            frames=[frame],
            onPage=self._aps_header_footer,
        )
        doc = _APSDocTemplate(
            buffer,
            pagesize=self._page_size,
            leftMargin=self._margin,
            rightMargin=self._margin,
            topMargin=self._margin + 0.3 * inch,
            bottomMargin=self._margin + 0.3 * inch,
        )
        doc.addPageTemplates([template])

        story = self._build_aps_story(summary, **kwargs)
        doc.multiBuild(story)
        return buffer.getvalue()

    # ── Style setup ──────────────────────────────────────────────────

    def _build_styles(self) -> dict[str, ParagraphStyle]:
        base = getSampleStyleSheet()
        font = self._config.font_family
        body_sz = self._config.body_font_size
        heading_sz = self._config.heading_font_size

        return {
            "title": ParagraphStyle(
                "title",
                parent=base["Title"],
                fontName=f"{font}-Bold",
                fontSize=heading_sz + 6,
                leading=(heading_sz + 6) * 1.2,
                alignment=TA_CENTER,
                spaceAfter=12,
            ),
            "heading": ParagraphStyle(
                "heading",
                parent=base["Heading2"],
                fontName=f"{font}-Bold",
                fontSize=heading_sz,
                leading=heading_sz * 1.3,
                spaceBefore=14,
                spaceAfter=6,
                textColor=_hex(HEADER_BG_COLOR),
            ),
            "body": ParagraphStyle(
                "body",
                parent=base["BodyText"],
                fontName=font,
                fontSize=body_sz,
                leading=body_sz * 1.4,
                spaceAfter=6,
            ),
            "bullet": ParagraphStyle(
                "bullet",
                parent=base["BodyText"],
                fontName=font,
                fontSize=body_sz,
                leading=body_sz * 1.4,
                leftIndent=18,
                bulletIndent=6,
                spaceAfter=3,
            ),
            "caption": ParagraphStyle(
                "caption",
                parent=base["BodyText"],
                fontName=f"{font}-Oblique",
                fontSize=body_sz - 1,
                textColor=rl_colors.grey,
                spaceAfter=4,
            ),
            "center": ParagraphStyle(
                "center",
                parent=base["BodyText"],
                fontName=font,
                fontSize=body_sz,
                alignment=TA_CENTER,
            ),
            "cover_subtitle": ParagraphStyle(
                "cover_subtitle",
                parent=base["BodyText"],
                fontName=font,
                fontSize=body_sz + 2,
                alignment=TA_CENTER,
                spaceAfter=6,
                textColor=rl_colors.grey,
            ),
            # ── APS-specific styles ──────────────────────────────
            "aps_heading_1": ParagraphStyle(
                "aps_heading_1",
                parent=base["Heading1"],
                fontName=f"{font}-Bold",
                fontSize=heading_sz + 2,
                leading=(heading_sz + 2) * 1.3,
                spaceBefore=16,
                spaceAfter=8,
                textColor=_hex(HEADER_BG_COLOR),
            ),
            "aps_heading_2": ParagraphStyle(
                "aps_heading_2",
                parent=base["Heading2"],
                fontName=f"{font}-Bold",
                fontSize=heading_sz,
                leading=heading_sz * 1.3,
                spaceBefore=12,
                spaceAfter=6,
                textColor=_hex(HEADER_BG_COLOR),
            ),
            "section_number": ParagraphStyle(
                "section_number",
                parent=base["BodyText"],
                fontName=f"{font}-Bold",
                fontSize=body_sz,
                textColor=_hex(HEADER_BG_COLOR),
            ),
            "citation_ref": ParagraphStyle(
                "citation_ref",
                parent=base["BodyText"],
                fontName=f"{font}-Oblique",
                fontSize=body_sz - 2,
                textColor=_hex(CITATION_TEXT_COLOR),
                spaceAfter=2,
            ),
            "demographics_label": ParagraphStyle(
                "demographics_label",
                parent=base["BodyText"],
                fontName=f"{font}-Bold",
                fontSize=body_sz,
                textColor=_hex(HEADER_BG_COLOR),
            ),
            "demographics_value": ParagraphStyle(
                "demographics_value",
                parent=base["BodyText"],
                fontName=font,
                fontSize=body_sz,
            ),
            "table_header": ParagraphStyle(
                "table_header",
                parent=base["BodyText"],
                fontName=f"{font}-Bold",
                fontSize=body_sz - 1,
                textColor=_hex(HEADER_TEXT_COLOR),
            ),
            "red_flag_text": ParagraphStyle(
                "red_flag_text",
                parent=base["BodyText"],
                fontName=f"{font}-Bold",
                fontSize=body_sz,
                textColor=_hex(RED_FLAG_BORDER_COLOR),
            ),
        }

    # ══════════════════════════════════════════════════════════════════
    # Legacy story construction (unchanged)
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_citation_index(
        batch_results: list[Any],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build a category -> citation list lookup from batch extraction results."""
        index: dict[str, list[dict[str, Any]]] = {}
        for br in batch_results:
            cat = br.category if hasattr(br, "category") else br.get("category", "")
            cits: list[dict[str, Any]] = []
            extractions = br.extractions if hasattr(br, "extractions") else br.get("extractions", [])
            for ext in extractions:
                ext_cits = ext.citations if hasattr(ext, "citations") else ext.get("citations", [])
                for c in ext_cits:
                    page = c.page_number if hasattr(c, "page_number") else c.get("page_number", 0)
                    quote = c.verbatim_quote if hasattr(c, "verbatim_quote") else c.get("verbatim_quote", "")
                    if page:
                        cits.append({"page": page, "quote": quote})
            if cits:
                index[cat] = cits
        return index

    def _build_story(self, summary: UnderwriterSummary, **kwargs: Any) -> list[Flowable]:
        story: list[Flowable] = []

        # Build citation index from batch_results if available
        citation_index: dict[str, list[dict[str, Any]]] = {}
        if kwargs.get("batch_results"):
            citation_index = self._build_citation_index(kwargs["batch_results"])

        if self._config.include_cover_page:
            story.extend(self._build_cover_page(summary))
        story.extend(self._build_executive_summary(summary))
        story.extend(self._build_risk_classification(summary))
        for section in summary.sections:
            story.extend(self._build_section(section, citation_index))
        story.extend(self._build_risk_factors(summary))
        story.extend(self._build_overall_assessment(summary))
        if self._config.include_appendix and kwargs.get("batch_results"):
            story.extend(self._build_appendix(kwargs["batch_results"]))
        return story

    def _build_cover_page(self, summary: UnderwriterSummary) -> list[Flowable]:
        items: list[Flowable] = [Spacer(1, 2 * inch)]
        items.append(Paragraph("APS Underwriter Summary", self._styles["title"]))
        items.append(Spacer(1, 0.3 * inch))
        if self._config.company_name:
            items.append(Paragraph(self._config.company_name, self._styles["cover_subtitle"]))
        items.append(Paragraph(f"Document: {summary.document_id}", self._styles["center"]))
        items.append(Spacer(1, 0.2 * inch))
        items.append(Paragraph(f"Patient: {summary.patient_demographics}", self._styles["center"]))
        items.append(Spacer(1, 0.15 * inch))
        items.append(Paragraph(f"Generated: {summary.generated_at}", self._styles["center"]))
        if self._config.confidential_watermark:
            items.append(Spacer(1, 0.5 * inch))
            items.append(
                Paragraph(
                    "<b>CONFIDENTIAL</b> — For authorized underwriting use only",
                    self._styles["center"],
                )
            )
        items.append(PageBreak())
        return items

    def _build_executive_summary(self, summary: UnderwriterSummary) -> list[Flowable]:
        items: list[Flowable] = [
            Paragraph("Executive Summary", self._styles["heading"]),
        ]
        items.append(
            Paragraph(
                f"Patient: {summary.patient_demographics}",
                self._styles["body"],
            )
        )
        total = summary.total_questions_answered
        high = summary.high_confidence_count
        pct = (high / total * 100) if total else 0
        items.append(
            Paragraph(
                f"Coverage: {total} questions answered, {high} at high confidence ({pct:.0f}%)",
                self._styles["body"],
            )
        )
        items.append(Spacer(1, 6))
        for section in summary.sections[:3]:
            for finding in section.key_findings[:2]:
                items.append(
                    Paragraph(f"\u2022 {finding}", self._styles["bullet"])
                )
        items.append(Spacer(1, 8))
        return items

    def _build_risk_classification(self, summary: UnderwriterSummary) -> list[Flowable]:
        items: list[Flowable] = [
            Paragraph("Risk Classification", self._styles["heading"]),
        ]
        risk_text = "See Overall Assessment"
        if summary.risk_factors:
            risk_text = f"{len(summary.risk_factors)} risk factor(s) identified"

        box_data = [[Paragraph(risk_text, self._styles["body"])]]
        box = Table(box_data, colWidths=[self._content_width()])
        box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), _hex(RISK_BOX_BG_COLOR)),
                    ("BOX", (0, 0), (-1, -1), 1, _hex(SECTION_BORDER_COLOR)),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )
        items.append(box)
        items.append(Spacer(1, 10))
        return items

    def _build_section(
        self,
        section: SynthesisSection,
        citation_index: dict[str, list[dict[str, Any]]] | None = None,
    ) -> list[Flowable]:
        items: list[Flowable] = [
            Paragraph(section.title, self._styles["heading"]),
            Paragraph(section.content, self._styles["body"]),
        ]
        for finding in section.key_findings:
            severity_prefix = self._detect_severity(finding)
            color = SEVERITY_COLORS.get(severity_prefix, SEVERITY_COLORS["INFORMATIONAL"])
            bullet = (
                f'<font color="{color}">\u25cf</font> {finding}'
            )
            items.append(Paragraph(bullet, self._styles["bullet"]))

        # Collect page citations for this section's source categories
        cited_pages: set[int] = set()
        if citation_index and section.source_categories:
            for cat in section.source_categories:
                for cit in citation_index.get(cat, []):
                    cited_pages.add(cit["page"])

        if section.source_categories:
            cats = ", ".join(
                CATEGORY_DISPLAY_NAMES.get(c, c) for c in section.source_categories
            )
            if cited_pages:
                page_refs = ", ".join(str(p) for p in sorted(cited_pages))
                items.append(Paragraph(
                    f"Sources: {cats} | Pages: {page_refs}",
                    self._styles["caption"],
                ))
            else:
                items.append(Paragraph(f"Sources: {cats}", self._styles["caption"]))
        items.append(Spacer(1, 6))
        return items

    def _build_risk_factors(self, summary: UnderwriterSummary) -> list[Flowable]:
        if not summary.risk_factors:
            return []
        items: list[Flowable] = [
            Paragraph("Risk Factors", self._styles["heading"]),
        ]
        for i, factor in enumerate(summary.risk_factors, 1):
            severity = self._detect_severity(factor)
            color = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["MODERATE"])
            text = f'<font color="{color}"><b>{i}.</b></font> {factor}'
            items.append(Paragraph(text, self._styles["body"]))
        items.append(Spacer(1, 10))
        return items

    def _build_overall_assessment(self, summary: UnderwriterSummary) -> list[Flowable]:
        if not summary.overall_assessment:
            return []
        items: list[Flowable] = [
            Paragraph("Overall Assessment", self._styles["heading"]),
        ]
        box_data = [[Paragraph(summary.overall_assessment, self._styles["body"])]]
        box = Table(box_data, colWidths=[self._content_width()])
        box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), _hex(ASSESSMENT_BOX_BG_COLOR)),
                    ("BOX", (0, 0), (-1, -1), 1.5, _hex(HEADER_BG_COLOR)),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )
        items.append(box)
        items.append(Spacer(1, 12))
        return items

    def _build_appendix(self, batch_results: list[Any]) -> list[Flowable]:
        items: list[Flowable] = [
            PageBreak(),
            Paragraph("Appendix — Detailed Extraction Results", self._styles["heading"]),
            Spacer(1, 8),
        ]
        header = ["Question ID", "Answer", "Confidence", "Pages"]
        header_row = [Paragraph(f"<b>{h}</b>", self._styles["body"]) for h in header]
        rows: list[list[Any]] = [header_row]

        for br in batch_results:
            extractions = getattr(br, "extractions", [])
            for er in extractions:
                pages = ", ".join(str(p) for p in getattr(er, "source_pages", []))
                conf = getattr(er, "confidence", 0)
                rows.append([
                    Paragraph(str(getattr(er, "question_id", "")), self._styles["body"]),
                    Paragraph(str(getattr(er, "answer", ""))[:200], self._styles["body"]),
                    Paragraph(f"{conf:.0%}", self._styles["body"]),
                    Paragraph(pages, self._styles["body"]),
                ])

        if len(rows) > 1:
            col_widths = [
                self._content_width() * 0.18,
                self._content_width() * 0.50,
                self._content_width() * 0.14,
                self._content_width() * 0.18,
            ]
            table = Table(rows, colWidths=col_widths, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), _hex(HEADER_BG_COLOR)),
                        ("TEXTCOLOR", (0, 0), (-1, 0), _hex(HEADER_TEXT_COLOR)),
                        ("GRID", (0, 0), (-1, -1), 0.5, _hex(SECTION_BORDER_COLOR)),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, HexColor("#F8FAFC")]),
                    ]
                )
            )
            items.append(table)
        else:
            items.append(Paragraph("No extraction results available.", self._styles["body"]))
        return items

    # ══════════════════════════════════════════════════════════════════
    # APS story construction (new)
    # ══════════════════════════════════════════════════════════════════

    def _build_aps_story(self, summary: APSSummary, **kwargs: Any) -> list[Flowable]:
        """Orchestrate the full APS PDF: cover → TOC → exec summary → sections → assessment."""
        story: list[Flowable] = []

        # Cover page
        if self._config.include_cover_page:
            story.extend(self._build_aps_cover_page(summary))

        # Table of Contents
        if self._config.include_toc:
            story.extend(self._build_toc())
            story.append(PageBreak())

        # Executive Summary
        story.extend(self._build_aps_executive_summary(summary))

        # Numbered sections
        section_num = 1
        for section in summary.sections:
            num_str = f"{section_num}.0" if self._config.section_numbering else ""
            story.extend(self._build_aps_section(section, num_str))
            section_num += 1

        # Red Flags
        if summary.red_flags and self._config.red_flag_alerts:
            story.extend(self._build_red_flags_section(summary.red_flags))

        # Overall Assessment
        if summary.overall_assessment:
            story.extend(self._build_aps_overall_assessment(summary.overall_assessment))

        # Appendix — citation index
        if self._config.include_appendix and summary.citation_index:
            story.extend(self._build_aps_appendix(summary))

        return story

    # ── APS Cover page ───────────────────────────────────────────────

    def _build_aps_cover_page(self, summary: APSSummary) -> list[Flowable]:
        items: list[Flowable] = [Spacer(1, 2 * inch)]
        items.append(Paragraph("APS Underwriter Summary", self._styles["title"]))
        items.append(Spacer(1, 0.3 * inch))
        if self._config.company_name:
            items.append(Paragraph(self._config.company_name, self._styles["cover_subtitle"]))
        items.append(Paragraph(f"Document: {summary.document_id}", self._styles["center"]))
        items.append(Spacer(1, 0.2 * inch))
        items.append(
            Paragraph(f"Patient: {summary.demographics.summary_text()}", self._styles["center"])
        )
        items.append(Spacer(1, 0.15 * inch))
        items.append(Paragraph(f"Generated: {summary.generated_at}", self._styles["center"]))
        if self._config.confidential_watermark:
            items.append(Spacer(1, 0.5 * inch))
            items.append(
                Paragraph(
                    "<b>CONFIDENTIAL</b> — For authorized underwriting use only",
                    self._styles["center"],
                )
            )
        items.append(PageBreak())
        return items

    # ── Table of Contents ────────────────────────────────────────────

    def _build_toc(self) -> list[Flowable]:
        items: list[Flowable] = [
            Paragraph("Table of Contents", self._styles["aps_heading_1"]),
            Spacer(1, 12),
        ]
        toc = TableOfContents()
        toc.levelStyles = [
            ParagraphStyle(
                "toc_level_0",
                fontName=f"{self._config.font_family}-Bold",
                fontSize=self._config.body_font_size,
                leading=self._config.body_font_size * 1.8,
                leftIndent=0,
                textColor=_hex(HEADER_BG_COLOR),
            ),
            ParagraphStyle(
                "toc_level_1",
                fontName=self._config.font_family,
                fontSize=self._config.body_font_size - 1,
                leading=(self._config.body_font_size - 1) * 1.6,
                leftIndent=20,
            ),
        ]
        items.append(toc)
        return items

    # ── APS Executive Summary ────────────────────────────────────────

    def _build_aps_executive_summary(self, summary: APSSummary) -> list[Flowable]:
        items: list[Flowable] = [
            Paragraph("Executive Summary", self._styles["aps_heading_1"]),
        ]

        # Demographics grid
        items.extend(self._build_demographics_grid(summary.demographics))
        items.append(Spacer(1, 8))

        # Risk badge
        if summary.risk_classification and summary.risk_classification.tier and self._config.risk_badge_enabled:
            items.extend(self._build_risk_badge(summary.risk_classification))
            items.append(Spacer(1, 8))

        # Coverage stats
        total = summary.total_questions_answered
        high = summary.high_confidence_count
        pct = (high / total * 100) if total else 0
        items.append(
            Paragraph(
                f"Coverage: {total} questions answered, {high} at high confidence ({pct:.0f}%)",
                self._styles["body"],
            )
        )

        # Top findings from first 3 sections
        items.append(Spacer(1, 6))
        for section in summary.sections[:3]:
            for finding in section.findings[:2]:
                items.extend(self._build_finding_with_citations(finding))

        # Top red flags
        if summary.red_flags:
            items.append(Spacer(1, 4))
            for rf in summary.red_flags[:3]:
                color = SEVERITY_COLORS.get(rf.severity, SEVERITY_COLORS["MODERATE"])
                items.append(
                    Paragraph(
                        f'<font color="{color}">\u26a0</font> {rf.description}',
                        self._styles["bullet"],
                    )
                )

        items.append(Spacer(1, 12))
        return items

    # ── Demographics grid ────────────────────────────────────────────

    def _build_demographics_grid(self, demographics: PatientDemographics) -> list[Flowable]:
        """2-column key-value table for patient demographics."""
        if not demographics.full_name and demographics.raw_text:
            return [Paragraph(f"Patient: {demographics.raw_text}", self._styles["body"])]

        pairs: list[tuple[str, str]] = []
        if demographics.full_name:
            pairs.append(("Name", demographics.full_name))
        if demographics.date_of_birth:
            pairs.append(("DOB", demographics.date_of_birth))
        if demographics.age:
            pairs.append(("Age", demographics.age))
        if demographics.gender:
            pairs.append(("Gender", demographics.gender))
        if demographics.insurance_id:
            pairs.append(("Insurance ID", demographics.insurance_id))
        if demographics.employer:
            pairs.append(("Employer", demographics.employer))
        if demographics.occupation:
            pairs.append(("Occupation", demographics.occupation))
        if demographics.phone:
            pairs.append(("Phone", demographics.phone))
        if demographics.address:
            pairs.append(("Address", demographics.address))

        if not pairs:
            return []

        # Build 2-column grid (label: value | label: value)
        rows: list[list[Any]] = []
        for i in range(0, len(pairs), 2):
            row: list[Any] = [
                Paragraph(f"<b>{pairs[i][0]}:</b>", self._styles["demographics_label"]),
                Paragraph(pairs[i][1], self._styles["demographics_value"]),
            ]
            if i + 1 < len(pairs):
                row.append(Paragraph(f"<b>{pairs[i + 1][0]}:</b>", self._styles["demographics_label"]))
                row.append(Paragraph(pairs[i + 1][1], self._styles["demographics_value"]))
            else:
                row.extend(["", ""])
            rows.append(row)

        cw = self._content_width()
        col_widths = [cw * 0.15, cw * 0.35, cw * 0.15, cw * 0.35]
        table = Table(rows, colWidths=col_widths)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), _hex(DEMOGRAPHICS_HEADER_BG)),
                    ("BOX", (0, 0), (-1, -1), 0.5, _hex(SECTION_BORDER_COLOR)),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, _hex(SECTION_BORDER_COLOR)),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return [table]

    # ── APS Section rendering ────────────────────────────────────────

    def _build_aps_section(self, section: APSSection, number: str) -> list[Flowable]:
        """Render a numbered APS section with narrative, findings, and typed tables."""
        items: list[Flowable] = []

        # Section header
        title = section.title or APS_SECTION_TITLES.get(section.section_key, section.section_key)
        header_text = f"{number} {title}" if number else title
        items.append(Paragraph(header_text, self._styles["aps_heading_1"]))

        # Narrative content
        if section.content:
            items.append(Paragraph(section.content, self._styles["body"]))

        # Findings with citations
        for finding in section.findings:
            items.extend(self._build_finding_with_citations(finding))

        # Typed data tables
        if section.conditions:
            items.extend(self._build_conditions_table(section.conditions))
        if section.lab_results:
            items.extend(self._build_lab_table(section.lab_results))
        if section.medications:
            items.extend(self._build_medication_table(section.medications))
        if section.encounters:
            items.extend(self._build_encounter_timeline(section.encounters))
        if section.vital_signs:
            items.extend(self._build_vital_signs_grid(section.vital_signs))
        if section.allergies:
            items.extend(self._build_allergy_table(section.allergies))
        if section.imaging_results:
            items.extend(self._build_imaging_table(section.imaging_results))
        if section.surgical_history:
            items.extend(self._build_surgical_table(section.surgical_history))

        # Source bar
        if section.source_categories:
            cats = ", ".join(
                CATEGORY_DISPLAY_NAMES.get(c, c) for c in section.source_categories
            )
            items.append(Paragraph(f"Sources: {cats}", self._styles["caption"]))

        items.append(Spacer(1, 8))
        return items

    # ── Finding with citations ───────────────────────────────────────

    def _build_finding_with_citations(self, finding: Finding) -> list[Flowable]:
        """Severity bullet + text + gray citation references."""
        items: list[Flowable] = []
        color = SEVERITY_COLORS.get(finding.severity, SEVERITY_COLORS["INFORMATIONAL"])
        bullet = f'<font color="{color}">\u25cf</font> {finding.text}'
        items.append(Paragraph(bullet, self._styles["bullet"]))

        if finding.citations and self._config.include_citation_refs:
            items.extend(self._build_citation_bar(finding.citations))
        return items

    # ── Citation bar ─────────────────────────────────────────────────

    def _build_citation_bar(self, citations: list[CitationRef]) -> list[Flowable]:
        """Gray italic citation references below a finding."""
        refs = "; ".join(c.display() for c in citations)
        return [Paragraph(refs, self._styles["citation_ref"])]

    # ── Lab table ────────────────────────────────────────────────────

    def _build_lab_table(self, results: list[Any]) -> list[Flowable]:
        """Render lab results with flag-colored cells and alternating rows."""
        items: list[Flowable] = [
            Spacer(1, 4),
            Paragraph("<b>Laboratory Results</b>", self._styles["body"]),
        ]
        header = ["Test", "Value", "Range", "Flag", "Date"]
        header_row = [Paragraph(f"<b>{h}</b>", self._styles["table_header"]) for h in header]
        rows: list[list[Any]] = [header_row]

        for lr in results:
            flag_color = LAB_FLAG_COLORS.get(lr.flag, LAB_FLAG_COLORS[""])
            if self._config.lab_table_flag_colors:
                flag_text = f'<font color="{flag_color}"><b>{lr.flag or "N"}</b></font>'
            else:
                flag_text = lr.flag or "N"
            rows.append([
                Paragraph(lr.test_name, self._styles["body"]),
                Paragraph(f"{lr.value} {lr.unit}".strip(), self._styles["body"]),
                Paragraph(lr.reference_range, self._styles["body"]),
                Paragraph(flag_text, self._styles["body"]),
                Paragraph(lr.date, self._styles["body"]),
            ])

        cw = self._content_width()
        col_widths = [cw * 0.25, cw * 0.20, cw * 0.25, cw * 0.10, cw * 0.20]
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _hex(HEADER_BG_COLOR)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), _hex(HEADER_TEXT_COLOR)),
                    ("GRID", (0, 0), (-1, -1), 0.5, _hex(SECTION_BORDER_COLOR)),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, HexColor("#F8FAFC")]),
                ]
            )
        )
        items.append(table)
        items.append(Spacer(1, 6))
        return items

    # ── Medication table ─────────────────────────────────────────────

    def _build_medication_table(self, meds: list[Any]) -> list[Flowable]:
        items: list[Flowable] = [
            Spacer(1, 4),
            Paragraph("<b>Medications</b>", self._styles["body"]),
        ]
        header = ["Name", "Dose", "Frequency", "Route", "Prescriber"]
        header_row = [Paragraph(f"<b>{h}</b>", self._styles["table_header"]) for h in header]
        rows: list[list[Any]] = [header_row]

        for med in meds:
            rows.append([
                Paragraph(med.name, self._styles["body"]),
                Paragraph(med.dose, self._styles["body"]),
                Paragraph(med.frequency, self._styles["body"]),
                Paragraph(med.route, self._styles["body"]),
                Paragraph(med.prescriber, self._styles["body"]),
            ])

        cw = self._content_width()
        col_widths = [cw * 0.25, cw * 0.20, cw * 0.20, cw * 0.15, cw * 0.20]
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _hex(HEADER_BG_COLOR)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), _hex(HEADER_TEXT_COLOR)),
                    ("GRID", (0, 0), (-1, -1), 0.5, _hex(SECTION_BORDER_COLOR)),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, HexColor("#F8FAFC")]),
                ]
            )
        )
        items.append(table)
        items.append(Spacer(1, 6))
        return items

    # ── Encounter timeline ───────────────────────────────────────────

    def _build_encounter_timeline(self, encounters: list[Any]) -> list[Flowable]:
        items: list[Flowable] = [
            Spacer(1, 4),
            Paragraph("<b>Encounter Chronology</b>", self._styles["body"]),
        ]
        header = ["Date", "Provider", "Type", "Summary"]
        header_row = [Paragraph(f"<b>{h}</b>", self._styles["table_header"]) for h in header]
        rows: list[list[Any]] = [header_row]

        sorted_encounters = sorted(encounters, key=lambda e: e.date)
        for enc in sorted_encounters:
            rows.append([
                Paragraph(enc.date, self._styles["body"]),
                Paragraph(enc.provider, self._styles["body"]),
                Paragraph(enc.encounter_type, self._styles["body"]),
                Paragraph(enc.summary, self._styles["body"]),
            ])

        cw = self._content_width()
        col_widths = [cw * 0.15, cw * 0.20, cw * 0.15, cw * 0.50]
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _hex(HEADER_BG_COLOR)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), _hex(HEADER_TEXT_COLOR)),
                    ("GRID", (0, 0), (-1, -1), 0.5, _hex(SECTION_BORDER_COLOR)),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, HexColor("#F8FAFC")]),
                ]
            )
        )
        items.append(table)
        items.append(Spacer(1, 6))
        return items

    # ── Vital signs grid ─────────────────────────────────────────────

    def _build_vital_signs_grid(self, vitals: list[Any]) -> list[Flowable]:
        items: list[Flowable] = [
            Spacer(1, 4),
            Paragraph("<b>Vital Signs</b>", self._styles["body"]),
        ]
        header = ["Name", "Value", "Date"]
        header_row = [Paragraph(f"<b>{h}</b>", self._styles["table_header"]) for h in header]
        rows: list[list[Any]] = [header_row]

        for vs in vitals:
            rows.append([
                Paragraph(vs.name, self._styles["body"]),
                Paragraph(vs.value, self._styles["body"]),
                Paragraph(vs.date, self._styles["body"]),
            ])

        cw = self._content_width()
        col_widths = [cw * 0.33, cw * 0.34, cw * 0.33]
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _hex(HEADER_BG_COLOR)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), _hex(HEADER_TEXT_COLOR)),
                    ("GRID", (0, 0), (-1, -1), 0.5, _hex(SECTION_BORDER_COLOR)),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, HexColor("#F8FAFC")]),
                ]
            )
        )
        items.append(table)
        items.append(Spacer(1, 6))
        return items

    # ── Allergy table ────────────────────────────────────────────────

    def _build_allergy_table(self, allergies: list[Any]) -> list[Flowable]:
        items: list[Flowable] = [
            Spacer(1, 4),
            Paragraph("<b>Allergies</b>", self._styles["body"]),
        ]
        header = ["Allergen", "Reaction", "Severity"]
        header_row = [Paragraph(f"<b>{h}</b>", self._styles["table_header"]) for h in header]
        rows: list[list[Any]] = [header_row]

        for a in allergies:
            rows.append([
                Paragraph(a.allergen, self._styles["body"]),
                Paragraph(a.reaction, self._styles["body"]),
                Paragraph(a.severity, self._styles["body"]),
            ])

        cw = self._content_width()
        col_widths = [cw * 0.40, cw * 0.35, cw * 0.25]
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _hex(HEADER_BG_COLOR)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), _hex(HEADER_TEXT_COLOR)),
                    ("GRID", (0, 0), (-1, -1), 0.5, _hex(SECTION_BORDER_COLOR)),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        items.append(table)
        items.append(Spacer(1, 6))
        return items

    # ── Conditions table ─────────────────────────────────────────────

    def _build_conditions_table(self, conditions: list[Any]) -> list[Flowable]:
        items: list[Flowable] = [
            Spacer(1, 4),
            Paragraph("<b>Conditions</b>", self._styles["body"]),
        ]
        header = ["Condition", "ICD-10", "Onset", "Status", "Severity"]
        header_row = [Paragraph(f"<b>{h}</b>", self._styles["table_header"]) for h in header]
        rows: list[list[Any]] = [header_row]

        for c in conditions:
            rows.append([
                Paragraph(c.name, self._styles["body"]),
                Paragraph(c.icd10_code, self._styles["body"]),
                Paragraph(c.onset_date, self._styles["body"]),
                Paragraph(c.status, self._styles["body"]),
                Paragraph(c.severity, self._styles["body"]),
            ])

        cw = self._content_width()
        col_widths = [cw * 0.30, cw * 0.15, cw * 0.15, cw * 0.15, cw * 0.25]
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _hex(HEADER_BG_COLOR)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), _hex(HEADER_TEXT_COLOR)),
                    ("GRID", (0, 0), (-1, -1), 0.5, _hex(SECTION_BORDER_COLOR)),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, HexColor("#F8FAFC")]),
                ]
            )
        )
        items.append(table)
        items.append(Spacer(1, 6))
        return items

    # ── Imaging table ────────────────────────────────────────────────

    def _build_imaging_table(self, results: list[Any]) -> list[Flowable]:
        items: list[Flowable] = [
            Spacer(1, 4),
            Paragraph("<b>Imaging &amp; Diagnostics</b>", self._styles["body"]),
        ]
        header = ["Modality", "Body Part", "Finding", "Date"]
        header_row = [Paragraph(f"<b>{h}</b>", self._styles["table_header"]) for h in header]
        rows: list[list[Any]] = [header_row]

        for ir in results:
            rows.append([
                Paragraph(ir.modality, self._styles["body"]),
                Paragraph(ir.body_part, self._styles["body"]),
                Paragraph(ir.finding or ir.impression, self._styles["body"]),
                Paragraph(ir.date, self._styles["body"]),
            ])

        cw = self._content_width()
        col_widths = [cw * 0.20, cw * 0.20, cw * 0.40, cw * 0.20]
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _hex(HEADER_BG_COLOR)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), _hex(HEADER_TEXT_COLOR)),
                    ("GRID", (0, 0), (-1, -1), 0.5, _hex(SECTION_BORDER_COLOR)),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        items.append(table)
        items.append(Spacer(1, 6))
        return items

    # ── Surgical history table ───────────────────────────────────────

    def _build_surgical_table(self, procedures: list[Any]) -> list[Flowable]:
        items: list[Flowable] = [
            Spacer(1, 4),
            Paragraph("<b>Surgical History</b>", self._styles["body"]),
        ]
        header = ["Procedure", "Date", "Outcome", "Complications"]
        header_row = [Paragraph(f"<b>{h}</b>", self._styles["table_header"]) for h in header]
        rows: list[list[Any]] = [header_row]

        for sh in procedures:
            rows.append([
                Paragraph(sh.procedure, self._styles["body"]),
                Paragraph(sh.date, self._styles["body"]),
                Paragraph(sh.outcome, self._styles["body"]),
                Paragraph(sh.complications, self._styles["body"]),
            ])

        cw = self._content_width()
        col_widths = [cw * 0.30, cw * 0.15, cw * 0.25, cw * 0.30]
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _hex(HEADER_BG_COLOR)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), _hex(HEADER_TEXT_COLOR)),
                    ("GRID", (0, 0), (-1, -1), 0.5, _hex(SECTION_BORDER_COLOR)),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        items.append(table)
        items.append(Spacer(1, 6))
        return items

    # ── Risk badge ───────────────────────────────────────────────────

    def _build_risk_badge(self, risk: RiskClassification) -> list[Flowable]:
        """Colored box with tier name, table rating, and rationale."""
        tier_color = RISK_TIER_COLORS.get(risk.tier, RISK_TIER_COLORS.get("Standard", "#2563EB"))
        items: list[Flowable] = []

        badge_content: list[list[Any]] = []
        tier_text = f'<font color="{tier_color}" size="14"><b>{risk.tier}</b></font>'
        if risk.table_rating:
            tier_text += f'  <font size="10">({risk.table_rating})</font>'
        badge_content.append([Paragraph(tier_text, self._styles["body"])])

        if risk.rationale:
            badge_content.append([Paragraph(risk.rationale, self._styles["body"])])

        if risk.debit_credits:
            badge_content.append([Paragraph(f"Adjustments: {risk.debit_credits}", self._styles["caption"])])

        box = Table(badge_content, colWidths=[self._content_width()])
        box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), _hex(RISK_BOX_BG_COLOR)),
                    ("BOX", (0, 0), (-1, -1), 2, _hex(tier_color)),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )
        items.append(box)
        return items

    # ── Red flags section ────────────────────────────────────────────

    def _build_red_flags_section(self, flags: list[RedFlag]) -> list[Flowable]:
        """Red-bordered alert boxes with severity and citations."""
        items: list[Flowable] = [
            Paragraph("Red Flags &amp; Alerts", self._styles["aps_heading_1"]),
        ]
        for rf in flags:
            severity_color = SEVERITY_COLORS.get(rf.severity, SEVERITY_COLORS["MODERATE"])
            content_parts: list[list[Any]] = [
                [Paragraph(
                    f'<font color="{severity_color}">\u26a0 {rf.severity}</font>: {rf.description}',
                    self._styles["red_flag_text"],
                )],
            ]
            if rf.category:
                content_parts.append([
                    Paragraph(f"Category: {rf.category}", self._styles["caption"]),
                ])
            if rf.citations and self._config.include_citation_refs:
                refs = "; ".join(c.display() for c in rf.citations)
                content_parts.append([Paragraph(refs, self._styles["citation_ref"])])

            box = Table(content_parts, colWidths=[self._content_width()])
            box.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), _hex(RED_FLAG_BG_COLOR)),
                        ("BOX", (0, 0), (-1, -1), 1.5, _hex(RED_FLAG_BORDER_COLOR)),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ]
                )
            )
            items.append(box)
            items.append(Spacer(1, 6))

        return items

    # ── APS Overall Assessment ───────────────────────────────────────

    def _build_aps_overall_assessment(self, text: str) -> list[Flowable]:
        items: list[Flowable] = [
            Paragraph("Overall Assessment", self._styles["aps_heading_1"]),
        ]
        box_data = [[Paragraph(text, self._styles["body"])]]
        box = Table(box_data, colWidths=[self._content_width()])
        box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), _hex(ASSESSMENT_BOX_BG_COLOR)),
                    ("BOX", (0, 0), (-1, -1), 1.5, _hex(HEADER_BG_COLOR)),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )
        items.append(box)
        items.append(Spacer(1, 12))
        return items

    # ── APS Appendix ─────────────────────────────────────────────────

    def _build_aps_appendix(self, summary: APSSummary) -> list[Flowable]:
        """Citation index by page + extraction stats."""
        items: list[Flowable] = [
            PageBreak(),
            Paragraph("Appendix — Citation Index", self._styles["aps_heading_1"]),
            Spacer(1, 8),
        ]

        header = ["Page", "Source Type", "Section", "Quote (excerpt)"]
        header_row = [Paragraph(f"<b>{h}</b>", self._styles["table_header"]) for h in header]
        rows: list[list[Any]] = [header_row]

        for page_num in sorted(summary.citation_index.keys()):
            for ref in summary.citation_index[page_num]:
                quote = ref.verbatim_quote
                quote_excerpt = quote[:120] + "..." if len(quote) > 120 else quote
                rows.append([
                    Paragraph(str(ref.page_number), self._styles["body"]),
                    Paragraph(ref.source_type, self._styles["body"]),
                    Paragraph(ref.section_title, self._styles["body"]),
                    Paragraph(quote_excerpt, self._styles["body"]),
                ])

        if len(rows) > 1:
            cw = self._content_width()
            col_widths = [cw * 0.08, cw * 0.17, cw * 0.25, cw * 0.50]
            table = Table(rows, colWidths=col_widths, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), _hex(HEADER_BG_COLOR)),
                        ("TEXTCOLOR", (0, 0), (-1, 0), _hex(HEADER_TEXT_COLOR)),
                        ("GRID", (0, 0), (-1, -1), 0.5, _hex(SECTION_BORDER_COLOR)),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, HexColor("#F8FAFC")]),
                    ]
                )
            )
            items.append(table)
        else:
            items.append(Paragraph("No citations recorded.", self._styles["body"]))

        return items

    # ── APS Header / footer ──────────────────────────────────────────

    def _aps_header_footer(self, canvas: Any, doc: Any) -> None:
        """Enhanced header/footer with page X of Y."""
        canvas.saveState()
        width, height = self._page_size

        # Header
        header_text = "CONFIDENTIAL \u2014 APS Underwriter Summary"
        if self._config.company_name:
            header_text = f"{self._config.company_name} | {header_text}"
        canvas.setFont(f"{self._config.font_family}-Bold", 8)
        canvas.setFillColor(_hex(HEADER_BG_COLOR))
        canvas.drawString(self._margin, height - self._margin + 6, header_text)

        # Footer
        canvas.setFont(self._config.font_family, 8)
        canvas.setFillColor(rl_colors.grey)
        page_num = f"Page {canvas.getPageNumber()}"
        canvas.drawString(self._margin, self._margin - 14, page_num)
        canvas.drawRightString(width - self._margin, self._margin - 14, "Generated by Scout AI by Ameritas")

        canvas.restoreState()

    # ── Legacy header / footer ───────────────────────────────────────

    def _header_footer(self, canvas: Any, doc: Any) -> None:
        canvas.saveState()
        width, height = self._page_size

        # Header
        header_text = "CONFIDENTIAL \u2014 APS Underwriter Summary"
        if self._config.company_name:
            header_text = f"{self._config.company_name} | {header_text}"
        canvas.setFont(f"{self._config.font_family}-Bold", 8)
        canvas.setFillColor(_hex(HEADER_BG_COLOR))
        canvas.drawString(self._margin, height - self._margin + 6, header_text)

        # Footer
        canvas.setFont(self._config.font_family, 8)
        canvas.setFillColor(rl_colors.grey)
        page_num = f"Page {canvas.getPageNumber()}"
        canvas.drawString(self._margin, self._margin - 14, page_num)
        canvas.drawRightString(width - self._margin, self._margin - 14, "Generated by Scout AI by Ameritas")

        canvas.restoreState()

    # ══════════════════════════════════════════════════════════════════
    # Underwriting template rendering
    # ══════════════════════════════════════════════════════════════════

    def _format_underwriting(self, summary: UnderwritingAPSSummary, **kwargs: Any) -> bytes:
        """Render ``UnderwritingAPSSummary`` to a professional underwriting PDF."""
        buffer = BytesIO()
        frame = Frame(
            self._margin,
            self._margin + 0.3 * inch,
            self._content_width(),
            float(self._page_size[1]) - 2 * self._margin - 0.6 * inch,
            id="main",
        )
        template = PageTemplate(
            id="main",
            frames=[frame],
            onPage=self._aps_header_footer,
        )
        doc = _APSDocTemplate(
            buffer,
            pagesize=self._page_size,
            leftMargin=self._margin,
            rightMargin=self._margin,
            topMargin=self._margin + 0.3 * inch,
            bottomMargin=self._margin + 0.3 * inch,
        )
        doc.addPageTemplates([template])

        story = self._build_uw_story(summary, **kwargs)
        doc.multiBuild(story)
        return buffer.getvalue()

    def _build_uw_story(self, summary: UnderwritingAPSSummary, **kwargs: Any) -> list[Flowable]:
        """Orchestrate the underwriting PDF story.

        Mirrors the full APS story structure (cover, TOC, executive summary,
        numbered sections, red flags, assessment, appendix) and inserts the
        underwriting-specific sections (demographics header, medical summary
        table, Y/N conditions grid, narrative sections) between the executive
        summary and the detailed APS sections.
        """
        story: list[Flowable] = []
        batch_results = kwargs.get("batch_results", [])

        # ── Cover page ───────────────────────────────────────────────
        if self._config.include_cover_page:
            story.extend(self._build_aps_cover_page(summary))

        # ── Table of Contents ────────────────────────────────────────
        if self._config.include_toc:
            story.extend(self._build_toc())
            story.append(PageBreak())

        # ── Executive Summary (demographics grid, risk badge, stats) ─
        story.extend(self._build_aps_executive_summary(summary))

        # ── Underwriting-specific sections ───────────────────────────
        # Demographics header (policy number, date range, total pages)
        story.extend(self._build_uw_demographics_header(summary))
        story.append(Spacer(1, 12))

        # Medical summary table
        if batch_results:
            story.extend(self._build_uw_medical_summary_table(summary, batch_results))
            story.append(Spacer(1, 12))

        # Y/N conditions grid
        if summary.yn_conditions:
            story.extend(self._build_uw_yn_conditions_grid(summary.yn_conditions))
            story.append(Spacer(1, 12))

        # Narrative sections
        narrative_sections = [
            ("Morbidity Concerns", summary.morbidity_concerns, summary.morbidity_citations),
            ("Mortality Concerns", summary.mortality_concerns, summary.mortality_citations),
            ("Residence & Travel", summary.residence_travel, summary.residence_citations),
        ]
        for title, content, citations in narrative_sections:
            if content:
                story.extend(self._build_uw_narrative_section(title, content, citations))
                story.append(Spacer(1, 8))

        # ── Numbered APS sections (same as standard APS PDF) ────────
        section_num = 1
        for section in summary.sections:
            num_str = f"{section_num}.0" if self._config.section_numbering else ""
            story.extend(self._build_aps_section(section, num_str))
            section_num += 1

        # ── Red Flags ────────────────────────────────────────────────
        if summary.red_flags and self._config.red_flag_alerts:
            story.extend(self._build_red_flags_section(summary.red_flags))

        # ── Overall Assessment ───────────────────────────────────────
        if summary.overall_assessment:
            story.extend(self._build_aps_overall_assessment(summary.overall_assessment))

        # ── Appendix — citation index ────────────────────────────────
        if self._config.include_appendix and summary.citation_index:
            story.extend(self._build_aps_appendix(summary))

        return story

    # ── Underwriting demographics header ─────────────────────────────

    def _build_uw_demographics_header(self, summary: UnderwritingAPSSummary) -> list[Flowable]:
        """2-row, 4-column demographics header for underwriting template."""
        items: list[Flowable] = [
            Paragraph("Underwriting APS Summary", self._styles["aps_heading_1"]),
        ]

        demo = summary.demographics
        pairs: list[tuple[str, str]] = [
            ("Name", demo.full_name or "N/A"),
            ("Policy Number", summary.policy_number or "N/A"),
            ("DOB", demo.date_of_birth or "N/A"),
            ("APS Date Range", summary.aps_date_range or "N/A"),
            ("Gender", demo.gender or "N/A"),
            ("Total Pages", str(summary.total_document_pages) if summary.total_document_pages else "N/A"),
        ]

        rows: list[list[Any]] = []
        for i in range(0, len(pairs), 2):
            row: list[Any] = [
                Paragraph(f"<b>{pairs[i][0]}:</b>", self._styles["demographics_label"]),
                Paragraph(pairs[i][1], self._styles["demographics_value"]),
            ]
            if i + 1 < len(pairs):
                row.append(Paragraph(f"<b>{pairs[i + 1][0]}:</b>", self._styles["demographics_label"]))
                row.append(Paragraph(pairs[i + 1][1], self._styles["demographics_value"]))
            else:
                row.extend(["", ""])
            rows.append(row)

        cw = self._content_width()
        col_widths = [cw * 0.15, cw * 0.35, cw * 0.18, cw * 0.32]
        table = Table(rows, colWidths=col_widths)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), _hex(DEMOGRAPHICS_HEADER_BG)),
                    ("BOX", (0, 0), (-1, -1), 0.5, _hex(SECTION_BORDER_COLOR)),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, _hex(SECTION_BORDER_COLOR)),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        items.append(table)
        return items

    # ── Underwriting medical summary table ───────────────────────────

    def _build_uw_medical_summary_table(
        self,
        summary: UnderwritingAPSSummary,
        batch_results: list[Any],
    ) -> list[Flowable]:
        """Multi-section summary table with value, date, and page columns."""
        items: list[Flowable] = [
            Paragraph("Medical Summary", self._styles["aps_heading_2"]),
        ]

        header = ["Field", "Value", "Date", "Page"]
        header_row = [Paragraph(f"<b>{h}</b>", self._styles["table_header"]) for h in header]
        rows: list[list[Any]] = [header_row]

        # Build answer lookup from batch results
        answer_map: dict[str, Any] = {}
        for br in batch_results:
            extractions = br.extractions if hasattr(br, "extractions") else br.get("extractions", [])
            for ext in extractions:
                qid = ext.question_id if hasattr(ext, "question_id") else ext.get("question_id", "")
                answer_map[qid] = ext

        # Summary fields to display
        fields = [
            ("Blood Pressure", "uw-vs-001"),
            ("BMI", "uw-vs-002"),
            ("Height/Weight", "uw-vs-003"),
            ("HbA1c", "uw-lab-003"),
            ("Lipid Panel", "uw-lab-004"),
            ("Most Recent Encounter", "uw-enc-002"),
            ("Total Encounters", "uw-enc-003"),
        ]

        for label, qid in fields:
            ext = answer_map.get(qid)
            if ext is None:
                continue
            answer = ext.answer if hasattr(ext, "answer") else ext.get("answer", "")
            if answer == "Not found":
                continue
            # Extract first page citation
            citations = ext.citations if hasattr(ext, "citations") else ext.get("citations", [])
            page_ref = ""
            if citations:
                c = citations[0]
                pn = c.page_number if hasattr(c, "page_number") else c.get("page_number", 0)
                if pn:
                    page_ref = f"p.{pn}"

            # Split date from answer text
            value_text, date_text = self._split_value_and_date(str(answer))

            rows.append([
                Paragraph(f"<b>{label}</b>", self._styles["body"]),
                Paragraph(value_text[:200], self._styles["body"]),
                Paragraph(date_text, self._styles["body"]),
                Paragraph(page_ref, self._styles["citation_ref"]),
            ])

        if len(rows) <= 1:
            return []

        cw = self._content_width()
        col_widths = [cw * 0.20, cw * 0.50, cw * 0.15, cw * 0.15]
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _hex(HEADER_BG_COLOR)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), _hex(HEADER_TEXT_COLOR)),
                    ("GRID", (0, 0), (-1, -1), 0.5, _hex(SECTION_BORDER_COLOR)),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, HexColor("#F8FAFC")]),
                ]
            )
        )
        items.append(table)
        return items

    # ── Y/N conditions grid ──────────────────────────────────────────

    def _build_uw_yn_conditions_grid(self, conditions: list[YNCondition]) -> list[Flowable]:
        """Color-coded Y/N conditions table for underwriting."""
        items: list[Flowable] = [
            Paragraph("Conditions Screening", self._styles["aps_heading_2"]),
        ]

        header = ["Condition", "Y/N", "Detailed Answer", "Page"]
        header_row = [Paragraph(f"<b>{h}</b>", self._styles["table_header"]) for h in header]
        rows: list[list[Any]] = [header_row]

        # Track row indices for Y/N coloring
        yn_cell_styles: list[tuple[int, str]] = []

        for cond in conditions:
            display_name = YN_CONDITION_DISPLAY_NAMES.get(cond.condition_name, cond.condition_name)
            if cond.time_qualifier and cond.time_qualifier not in display_name:
                display_name = f"{display_name} ({cond.time_qualifier})"

            yn_color = YN_COLORS.get(cond.answer_yn, YN_COLORS["Unknown"])
            yn_text = f'<font color="{yn_color}"><b>{cond.answer_yn}</b></font>'

            detail = cond.detail[:200] if cond.detail else ""
            page_ref = ""
            if cond.citations:
                pages = sorted({c.page_number for c in cond.citations if c.page_number})
                if pages:
                    page_ref = ", ".join(f"p.{p}" for p in pages[:3])

            row_idx = len(rows)
            yn_cell_styles.append((row_idx, cond.answer_yn))

            rows.append([
                Paragraph(display_name, self._styles["body"]),
                Paragraph(yn_text, self._styles["body"]),
                Paragraph(detail, self._styles["body"]),
                Paragraph(page_ref, self._styles["citation_ref"]),
            ])

        cw = self._content_width()
        col_widths = [cw * 0.28, cw * 0.07, cw * 0.50, cw * 0.15]
        table = Table(rows, colWidths=col_widths, repeatRows=1)

        base_style: list[Any] = [
            ("BACKGROUND", (0, 0), (-1, 0), _hex(HEADER_BG_COLOR)),
            ("TEXTCOLOR", (0, 0), (-1, 0), _hex(HEADER_TEXT_COLOR)),
            ("GRID", (0, 0), (-1, -1), 0.5, _hex(SECTION_BORDER_COLOR)),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, HexColor("#F8FAFC")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]

        # Color-code Y/N cells
        _yn_bg_map = {
            "Y": "#FEE2E2",       # light red background
            "N": "#DCFCE7",       # light green background
            "Unknown": "#FEF3C7",  # light amber background
        }
        for row_idx, yn_val in yn_cell_styles:
            bg = _yn_bg_map.get(yn_val, "#FFFFFF")
            base_style.append(("BACKGROUND", (1, row_idx), (1, row_idx), HexColor(bg)))

        table.setStyle(TableStyle(base_style))
        items.append(table)
        return items

    # ── Underwriting narrative section ───────────────────────────────

    def _build_uw_narrative_section(
        self,
        title: str,
        content: str,
        citations: list[CitationRef],
    ) -> list[Flowable]:
        """Reusable narrative section with page references."""
        items: list[Flowable] = [
            Paragraph(title, self._styles["aps_heading_2"]),
        ]

        box_data: list[list[Any]] = [
            [Paragraph(content, self._styles["body"])],
        ]
        if citations and self._config.include_citation_refs:
            refs = "; ".join(c.display() for c in citations)
            box_data.append([Paragraph(refs, self._styles["citation_ref"])])

        box = Table(box_data, colWidths=[self._content_width()])
        box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), _hex(ASSESSMENT_BOX_BG_COLOR)),
                    ("BOX", (0, 0), (-1, -1), 1, _hex(SECTION_BORDER_COLOR)),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        items.append(box)
        return items

    # ── Helpers ───────────────────────────────────────────────────────

    # Date pattern: MM/DD/YYYY or MM-DD-YYYY
    _DATE_RE = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")
    # Common prefixes that signal a date fragment: "date:", "dated", "on"
    _DATE_PREFIX_RE = re.compile(r",?\s*(?:date[d]?\s*:\s*|on\s+)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", re.IGNORECASE)

    @classmethod
    def _split_value_and_date(cls, answer: str) -> tuple[str, str]:
        """Extract the last date from *answer*, returning (value, date).

        Handles patterns like:
        - "BMI 28.3, date: 03/15/2024" -> ("BMI 28.3", "03/15/2024")
        - "HbA1c: 5.8%, date: 03/15/2024" -> ("HbA1c: 5.8%", "03/15/2024")
        - "12/15/2024 - Annual physical" -> ("Annual physical", "12/15/2024")
        - "12" -> ("12", "")
        """
        # Try explicit "date:" pattern first — strip the prefix+date from value
        m = cls._DATE_PREFIX_RE.search(answer)
        if m:
            date_str = m.group(1)
            value = answer[:m.start()].rstrip(", ")
            return value, date_str

        # Find all date-like tokens
        dates = cls._DATE_RE.findall(answer)
        if not dates:
            return answer, ""

        # If the answer starts with a date followed by a separator, treat it
        # as "date - description" (e.g., "12/15/2024 - Annual physical exam")
        leading = re.match(r"^(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*[-:]\s*(.+)", answer)
        if leading:
            return leading.group(2).strip(), leading.group(1)

        # Otherwise use the last date found and strip it from the value
        last_date = dates[-1]
        # Remove the date and surrounding separators
        value = re.sub(r",?\s*\(?" + re.escape(last_date) + r"\)?\s*,?", "", answer).strip().rstrip(",")
        return value, last_date

    def _content_width(self) -> float:
        return float(self._page_size[0]) - 2 * self._margin

    _SEVERITY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"\b(critical|severe|emergency|urgent)\b", re.IGNORECASE), "CRITICAL"),
        (re.compile(r"\b(significant|major|elevated\s+risk)\b", re.IGNORECASE), "SIGNIFICANT"),
        (re.compile(r"\b(moderate|borderline)\b", re.IGNORECASE), "MODERATE"),
        (re.compile(r"\b(minor|mild|low)\b", re.IGNORECASE), "MINOR"),
    ]

    @staticmethod
    def _detect_severity(text: str) -> str:
        """Keyword-based severity detection using word-boundary matching."""
        for pattern, level in PDFFormatter._SEVERITY_PATTERNS:
            if pattern.search(text):
                return level
        return "INFORMATIONAL"
