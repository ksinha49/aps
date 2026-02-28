"""PDF output formatter using reportlab.

Renders ``UnderwriterSummary`` as a professionally formatted APS summary PDF
suitable for underwriter review.  Requires the ``pdf`` optional dependency::

    pip install pageindex-rag[pdf]
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from pageindex_rag.core.config import PDFFormattingConfig
from pageindex_rag.formatters.pdf_styles import (
    ASSESSMENT_BOX_BG_COLOR,
    CATEGORY_DISPLAY_NAMES,
    HEADER_BG_COLOR,
    HEADER_TEXT_COLOR,
    RISK_BOX_BG_COLOR,
    SECTION_BORDER_COLOR,
    SEVERITY_COLORS,
)
from pageindex_rag.synthesis.models import SynthesisSection, UnderwriterSummary

try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4, LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Flowable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except ImportError as _exc:
    raise ImportError(
        "reportlab is required for PDF output. Install with: pip install pageindex-rag[pdf]"
    ) from _exc


# ── Page size lookup ─────────────────────────────────────────────────

_PAGE_SIZES = {"letter": LETTER, "a4": A4}


def _hex(color_str: str) -> HexColor:
    return HexColor(color_str)


# ── PDFFormatter ─────────────────────────────────────────────────────


class PDFFormatter:
    """Renders ``UnderwriterSummary`` as a professional APS summary PDF."""

    def __init__(self, config: PDFFormattingConfig | None = None) -> None:
        self._config = config or PDFFormattingConfig()
        self._page_size: tuple[float, float] = _PAGE_SIZES.get(self._config.page_size, LETTER)
        self._margin: float = self._config.margin_inches * inch
        self._styles = self._build_styles()

    # ── Public API ───────────────────────────────────────────────────

    def format(self, summary: UnderwriterSummary, **kwargs: Any) -> bytes:
        """Render *summary* to PDF bytes."""
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self._page_size,
            leftMargin=self._margin,
            rightMargin=self._margin,
            topMargin=self._margin + 0.3 * inch,  # room for header
            bottomMargin=self._margin + 0.3 * inch,
        )
        story = self._build_story(summary, **kwargs)
        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        return buffer.getvalue()

    def format_to_file(self, summary: UnderwriterSummary, path: Path, **kwargs: Any) -> Path:
        """Write PDF to *path* and return it."""
        path.write_bytes(self.format(summary, **kwargs))
        return path

    @property
    def content_type(self) -> str:
        return "application/pdf"

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
        }

    # ── Story construction ───────────────────────────────────────────

    def _build_story(self, summary: UnderwriterSummary, **kwargs: Any) -> list[Flowable]:
        story: list[Flowable] = []
        if self._config.include_cover_page:
            story.extend(self._build_cover_page(summary))
        story.extend(self._build_executive_summary(summary))
        story.extend(self._build_risk_classification(summary))
        for section in summary.sections:
            story.extend(self._build_section(section))
        story.extend(self._build_risk_factors(summary))
        story.extend(self._build_overall_assessment(summary))
        if self._config.include_appendix and kwargs.get("batch_results"):
            story.extend(self._build_appendix(kwargs["batch_results"]))
        return story

    # ── Cover page ───────────────────────────────────────────────────

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

    # ── Executive summary ────────────────────────────────────────────

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
        # Key findings from first few sections
        for section in summary.sections[:3]:
            for finding in section.key_findings[:2]:
                items.append(
                    Paragraph(f"\u2022 {finding}", self._styles["bullet"])
                )
        items.append(Spacer(1, 8))
        return items

    # ── Risk classification ──────────────────────────────────────────

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

    # ── Section rendering ────────────────────────────────────────────

    def _build_section(self, section: SynthesisSection) -> list[Flowable]:
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

        if section.source_categories:
            cats = ", ".join(
                CATEGORY_DISPLAY_NAMES.get(c, c) for c in section.source_categories
            )
            items.append(Paragraph(f"Sources: {cats}", self._styles["caption"]))
        items.append(Spacer(1, 6))
        return items

    # ── Risk factors ─────────────────────────────────────────────────

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

    # ── Overall assessment ───────────────────────────────────────────

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

    # ── Appendix ─────────────────────────────────────────────────────

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

    # ── Header / footer ──────────────────────────────────────────────

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
        canvas.drawRightString(width - self._margin, self._margin - 14, "Generated by PageIndex RAG")

        canvas.restoreState()

    # ── Helpers ───────────────────────────────────────────────────────

    def _content_width(self) -> float:
        return float(self._page_size[0]) - 2 * self._margin

    @staticmethod
    def _detect_severity(text: str) -> str:
        """Simple keyword-based severity detection from finding text."""
        lower = text.lower()
        if any(kw in lower for kw in ("critical", "severe", "emergency", "urgent")):
            return "CRITICAL"
        if any(kw in lower for kw in ("significant", "major", "elevated risk")):
            return "SIGNIFICANT"
        if any(kw in lower for kw in ("moderate", "borderline")):
            return "MODERATE"
        if any(kw in lower for kw in ("minor", "mild", "low")):
            return "MINOR"
        return "INFORMATIONAL"
