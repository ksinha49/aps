"""Extraction endpoint with optional synthesis and PDF export."""

from __future__ import annotations

import re
from io import BytesIO
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scout_ai.api.routes._prompt_context import PromptContextOverride, apply_prompt_context

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
    domain: str = "aps"
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


class ValidationIssueResponse(BaseModel):
    """A single validation issue in the response."""

    rule_id: str
    rule_name: str
    severity: str
    category: str
    message: str
    section_key: str = ""
    field_path: str = ""
    entity_name: str = ""
    actual_value: str = ""
    expected_hint: str = ""


class ValidationReportResponse(BaseModel):
    """Validation report included in extraction response."""

    document_id: str = ""
    total_rules_evaluated: int = 0
    total_issues: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    issues: list[ValidationIssueResponse] = Field(default_factory=list)
    passed: bool = True
    validated_at: str = ""
    rules_version: int = 1


class ExtractResponse(BaseModel):
    """Extraction results with optional synthesis and validation."""

    doc_id: str
    results: list[ExtractAnswer]
    summary: UnderwriterSummaryResponse | None = None
    validation: ValidationReportResponse | None = None


@router.post("/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest, req: Request) -> ExtractResponse:
    """Extract answers from a document using tiered extraction.

    When ``synthesize`` is True, additionally aggregates results into
    an underwriter summary.
    """
    settings = req.app.state.settings
    apply_prompt_context(request.prompt_context, settings.prompt)

    from scout_ai.models import ExtractionQuestion
    from scout_ai.services.extraction_service import ExtractionService
    from scout_ai.services.index_store import IndexStore

    store = IndexStore(base_path=settings.persistence.store_path)
    index = store.load(request.doc_id)

    questions = [
        ExtractionQuestion(
            question_id=q.question_id,
            category=q.category,
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
    validation_response: ValidationReportResponse | None = None
    if request.synthesize:
        from scout_ai.config import ScoutSettings
        from scout_ai.providers.pageindex.client import LLMClient
        from scout_ai.synthesis.pipeline import SynthesisPipeline
        from scout_ai.validation import create_rules_engine

        legacy_settings = ScoutSettings(
            llm_base_url=settings.llm.base_url,
            llm_api_key=settings.llm.api_key,
            llm_model=settings.llm.model,
            llm_temperature=settings.llm.temperature,
            llm_top_p=settings.llm.top_p,
            llm_seed=settings.llm.seed,
            llm_timeout=settings.llm.timeout,
            llm_max_retries=settings.llm.max_retries,
            retrieval_max_concurrent=settings.retrieval.max_concurrent,
        )
        client = LLMClient(legacy_settings)
        cache_enabled = settings.caching.enabled
        synth = SynthesisPipeline(client, cache_enabled=cache_enabled)
        metadata = {"doc_id": request.doc_id}

        # Create rules engine for post-LLM validation
        rules_engine = create_rules_engine(settings)

        # Use structured synthesis with validation
        aps_summary, validation_report = await synth.synthesize_structured(
            batch_results,
            metadata,
            rules_engine=rules_engine,
            total_pages=index.total_pages,
        )
        uw_summary = aps_summary.to_underwriter_summary()

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

        # Build validation response if available
        if validation_report is not None:
            validation_response = ValidationReportResponse(
                document_id=validation_report.document_id,
                total_rules_evaluated=validation_report.total_rules_evaluated,
                total_issues=validation_report.total_issues,
                error_count=validation_report.error_count,
                warning_count=validation_report.warning_count,
                info_count=validation_report.info_count,
                issues=[
                    ValidationIssueResponse(
                        rule_id=i.rule_id,
                        rule_name=i.rule_name,
                        severity=i.severity.value,
                        category=i.category.value,
                        message=i.message,
                        section_key=i.section_key,
                        field_path=i.field_path,
                        entity_name=i.entity_name,
                        actual_value=i.actual_value,
                        expected_hint=i.expected_hint,
                    )
                    for i in validation_report.issues
                ],
                passed=validation_report.passed,
                validated_at=validation_report.validated_at,
                rules_version=validation_report.rules_version,
            )

            # If fail_on_error is enabled and there are errors, reject
            if settings.rules.fail_on_error and validation_report.has_errors():
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "Validation failed with ERROR-level issues",
                        "error_count": validation_report.error_count,
                        "issues": [
                            {"rule_id": i.rule_id, "message": i.message}
                            for i in validation_report.issues
                            if i.severity.value == "error"
                        ],
                    },
                )

    return ExtractResponse(
        doc_id=request.doc_id,
        results=answers,
        summary=summary_response,
        validation=validation_response,
    )


# ── Export endpoint ──────────────────────────────────────────────────


class ExportRequest(BaseModel):
    """Request to export an underwriter summary in a given format."""

    summary: UnderwriterSummaryResponse
    output_format: Literal["pdf", "json"] = "pdf"


@router.post("/extract/export")
async def export_summary(request: ExportRequest, req: Request) -> StreamingResponse:
    """Export an underwriter summary as PDF or JSON."""
    from scout_ai.synthesis.models import SynthesisSection, UnderwriterSummary

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
            from scout_ai.formatters.pdf_formatter import PDFFormatter
        except ImportError as exc:
            raise HTTPException(
                status_code=501,
                detail="PDF export requires the 'pdf' extra: pip install scout-ai[pdf]",
            ) from exc
        formatter = PDFFormatter(req.app.state.settings.pdf)
    else:
        from scout_ai.formatters.json_formatter import JSONFormatter

        formatter = JSONFormatter()

    output_bytes = formatter.format(summary)
    safe_doc_id = re.sub(r"[^\w\-]", "_", request.summary.document_id)

    return StreamingResponse(
        BytesIO(output_bytes),
        media_type=formatter.content_type,
        headers={"Content-Disposition": f'attachment; filename="summary_{safe_doc_id}.{request.output_format}"'},
    )
