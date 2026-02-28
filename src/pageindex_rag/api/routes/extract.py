"""Extraction endpoint with optional synthesis and PDF export."""

from __future__ import annotations

import re
from io import BytesIO
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pageindex_rag.api.routes._prompt_context import PromptContextOverride, apply_prompt_context

router = APIRouter(tags=["extraction"])


class ExtractQuestion(BaseModel):
    """A single extraction question."""

    question_id: str
    category: str
    question_text: str
    tier: int = Field(default=1, ge=1, le=3)


class ExtractRequest(BaseModel):
    """Request to extract answers from a document."""

    doc_id: str
    questions: list[ExtractQuestion]
    synthesize: bool = False
    prompt_context: PromptContextOverride | None = None


class ExtractAnswer(BaseModel):
    """A single extracted answer."""

    question_id: str
    answer: str = ""
    confidence: float = 0.0
    source_pages: list[int] = Field(default_factory=list)


class SynthesisSectionResponse(BaseModel):
    """A section of the underwriter summary."""

    title: str
    content: str
    source_categories: list[str] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)


class UnderwriterSummaryResponse(BaseModel):
    """Underwriter summary produced from extraction results."""

    document_id: str
    patient_demographics: str = ""
    sections: list[SynthesisSectionResponse] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    overall_assessment: str = ""
    total_questions_answered: int = 0
    high_confidence_count: int = 0
    generated_at: str = ""


class ExtractResponse(BaseModel):
    """Extraction results with optional synthesis."""

    doc_id: str
    results: list[ExtractAnswer]
    summary: UnderwriterSummaryResponse | None = None


@router.post("/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest, req: Request) -> ExtractResponse:
    """Extract answers from a document using tiered extraction.

    When ``synthesize`` is True, additionally aggregates results into
    an underwriter summary.
    """
    settings = req.app.state.settings
    apply_prompt_context(request.prompt_context, settings.prompt)

    from pageindex_rag.models import ExtractionCategory, ExtractionQuestion
    from pageindex_rag.services.extraction_service import ExtractionService
    from pageindex_rag.services.index_store import IndexStore

    store = IndexStore(base_path=settings.persistence.store_path)
    index = store.load(request.doc_id)

    questions = [
        ExtractionQuestion(
            question_id=q.question_id,
            category=ExtractionCategory(q.category),
            question_text=q.question_text,
            tier=q.tier,
        )
        for q in request.questions
    ]

    service = ExtractionService(settings=settings)
    batch_results = await service.extract(index=index, questions=questions)

    answers = []
    for br in batch_results:
        for er in br.extractions:
            answers.append(
                ExtractAnswer(
                    question_id=er.question_id,
                    answer=er.answer,
                    confidence=er.confidence,
                    source_pages=er.source_pages,
                )
            )

    summary_response: UnderwriterSummaryResponse | None = None
    if request.synthesize:
        from pageindex_rag.config import PageIndexSettings
        from pageindex_rag.providers.pageindex.client import LLMClient
        from pageindex_rag.synthesis.pipeline import SynthesisPipeline

        legacy_settings = PageIndexSettings(
            llm_base_url=settings.llm.base_url,
            llm_api_key=settings.llm.api_key,
            llm_model=settings.llm.model,
            llm_temperature=settings.llm.temperature,
            llm_timeout=settings.llm.timeout,
            llm_max_retries=settings.llm.max_retries,
            retrieval_max_concurrent=settings.retrieval.max_concurrent,
        )
        client = LLMClient(legacy_settings)
        cache_enabled = settings.caching.enabled
        synth = SynthesisPipeline(client, cache_enabled=cache_enabled)
        metadata = {"doc_id": request.doc_id}
        uw_summary = await synth.synthesize(batch_results, metadata)

        summary_response = UnderwriterSummaryResponse(
            document_id=uw_summary.document_id,
            patient_demographics=uw_summary.patient_demographics,
            sections=[
                SynthesisSectionResponse(
                    title=s.title,
                    content=s.content,
                    source_categories=s.source_categories,
                    key_findings=s.key_findings,
                )
                for s in uw_summary.sections
            ],
            risk_factors=uw_summary.risk_factors,
            overall_assessment=uw_summary.overall_assessment,
            total_questions_answered=uw_summary.total_questions_answered,
            high_confidence_count=uw_summary.high_confidence_count,
            generated_at=uw_summary.generated_at,
        )

    return ExtractResponse(doc_id=request.doc_id, results=answers, summary=summary_response)


# ── Export endpoint ──────────────────────────────────────────────────


class ExportRequest(BaseModel):
    """Request to export an underwriter summary in a given format."""

    summary: UnderwriterSummaryResponse
    output_format: Literal["pdf", "json"] = "pdf"


@router.post("/extract/export")
async def export_summary(request: ExportRequest, req: Request) -> StreamingResponse:
    """Export an underwriter summary as PDF or JSON."""
    from pageindex_rag.synthesis.models import SynthesisSection, UnderwriterSummary

    summary = UnderwriterSummary(
        document_id=request.summary.document_id,
        patient_demographics=request.summary.patient_demographics,
        sections=[
            SynthesisSection(
                title=s.title,
                content=s.content,
                source_categories=s.source_categories,
                key_findings=s.key_findings,
            )
            for s in request.summary.sections
        ],
        risk_factors=request.summary.risk_factors,
        overall_assessment=request.summary.overall_assessment,
        total_questions_answered=request.summary.total_questions_answered,
        high_confidence_count=request.summary.high_confidence_count,
        generated_at=request.summary.generated_at,
    )

    if request.output_format == "pdf":
        try:
            from pageindex_rag.formatters.pdf_formatter import PDFFormatter
        except ImportError as exc:
            raise HTTPException(
                status_code=501,
                detail="PDF export requires the 'pdf' extra: pip install pageindex-rag[pdf]",
            ) from exc
        formatter = PDFFormatter(req.app.state.settings.pdf)
    else:
        from pageindex_rag.formatters.json_formatter import JSONFormatter

        formatter = JSONFormatter()

    output_bytes = formatter.format(summary)
    safe_doc_id = re.sub(r"[^\w\-]", "_", request.summary.document_id)

    return StreamingResponse(
        BytesIO(output_bytes),
        media_type=formatter.content_type,
        headers={"Content-Disposition": f'attachment; filename="aps_summary_{safe_doc_id}.{request.output_format}"'},
    )
