"""Synthesis pipeline: aggregates 900 extraction results into a structured underwriter summary."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from scout_ai.domains.aps.models import (
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
    SynthesisSection,
    UnderwriterSummary,
    UnderwritingAPSSummary,
    VitalSign,
    YNCondition,
)
from scout_ai.models import BatchExtractionResult
from scout_ai.prompts.registry import get_prompt
from scout_ai.providers.pageindex.client import LLMClient

if TYPE_CHECKING:
    from scout_ai.domains.aps.validation.engine import RulesEngine
    from scout_ai.validation.models import ValidationReport

log = logging.getLogger(__name__)


class SynthesisPipeline:
    """Aggregates extraction results into a structured underwriter summary.

    The pipeline works in two phases:
    1. **Prepare**: Filter to high-confidence answers, group by category
    2. **Generate**: LLM synthesizes category summaries into a narrative report

    When ``cache_enabled`` is True, the synthesis system prompt is cached
    for reuse across multiple documents processed in the same session.
    """

    def __init__(self, client: LLMClient, *, cache_enabled: bool = False) -> None:
        self._client = client
        self._cache_enabled = cache_enabled

    # ── Legacy synthesis (unchanged API) ────────────────────────────

    async def synthesize(
        self,
        extraction_results: list[BatchExtractionResult],
        document_metadata: dict[str, Any] | None = None,
    ) -> UnderwriterSummary:
        """Produce an underwriter summary from extraction results.

        Args:
            extraction_results: Per-category extraction results from the pipeline.
            document_metadata: Optional metadata (doc_id, doc_name, etc.).

        Returns:
            Structured ``UnderwriterSummary`` with sections, risk factors,
            and overall assessment.
        """
        metadata = document_metadata or {}

        # Phase 1: Prepare per-category summaries
        category_summaries = self._prepare_category_summaries(extraction_results)

        # Phase 2: Generate narrative via LLM
        system_prompt_text = get_prompt("aps", "synthesis", "SYNTHESIS_SYSTEM_PROMPT")
        synthesis_prompt_template = get_prompt("aps", "synthesis", "SYNTHESIS_PROMPT")

        system_prompt: str | None = system_prompt_text if self._cache_enabled else None

        prompt_body = synthesis_prompt_template.format(
            category_summaries=json.dumps(category_summaries, indent=2),
            document_metadata=json.dumps(metadata),
        )

        if not self._cache_enabled:
            # Inline system prompt into user message when caching is off
            prompt_body = f"{system_prompt_text}\n\n{prompt_body}"

        response = await self._client.complete(
            prompt_body,
            system_prompt=system_prompt,
            cache_system=self._cache_enabled,
        )

        return self._parse_summary(response, extraction_results, metadata)

    # ── Structured synthesis (APS Schema v1.0.0) ────────────────────

    async def synthesize_structured(
        self,
        extraction_results: list[BatchExtractionResult],
        document_metadata: dict[str, Any] | None = None,
        *,
        rules_engine: RulesEngine | None = None,
        total_pages: int = 0,
    ) -> tuple[APSSummary, ValidationReport | None]:
        """Produce a richly-typed APS summary from extraction results.

        Uses the structured prompt to request typed sections with findings,
        conditions, medications, lab results, and citation references.
        Falls back to flat parsing if the LLM returns a non-structured response.

        When ``rules_engine`` is provided, validates the parsed summary and
        returns a ``ValidationReport`` alongside the summary.

        Args:
            extraction_results: Per-category extraction results from the pipeline.
            document_metadata: Optional metadata (doc_id, doc_name, etc.).
            rules_engine: Optional validation engine for post-LLM checks.
            total_pages: Total document pages (for citation range validation).

        Returns:
            Tuple of (``APSSummary``, optional ``ValidationReport``).
        """
        metadata = document_metadata or {}

        # Phase 1: Prepare per-category summaries with full citations
        category_summaries = self._prepare_category_summaries(extraction_results)

        # Phase 2: Build citation index from raw extractions
        citation_index = self._build_citation_index(extraction_results)

        # Phase 3: Generate structured response via LLM
        system_prompt_text = get_prompt("aps", "synthesis", "SYNTHESIS_SYSTEM_PROMPT")
        synthesis_prompt_template = get_prompt("aps", "synthesis", "SYNTHESIS_STRUCTURED_PROMPT")

        system_prompt: str | None = system_prompt_text if self._cache_enabled else None

        prompt_body = synthesis_prompt_template.format(
            category_summaries=json.dumps(category_summaries, indent=2),
            document_metadata=json.dumps(metadata),
        )

        if not self._cache_enabled:
            prompt_body = f"{system_prompt_text}\n\n{prompt_body}"

        response = await self._client.complete(
            prompt_body,
            system_prompt=system_prompt,
            cache_system=self._cache_enabled,
        )

        aps_summary = self._parse_aps_summary(
            response, extraction_results, metadata, citation_index,
        )

        validation_report: ValidationReport | None = None
        if rules_engine is not None:
            validation_report = rules_engine.validate(
                aps_summary, total_pages=total_pages,
            )

        return aps_summary, validation_report

    # ── Category summary preparation ────────────────────────────────

    def _prepare_category_summaries(
        self,
        results: list[BatchExtractionResult],
    ) -> list[dict[str, Any]]:
        """Filter to high-confidence answers, group by category.

        Preserves full citation objects (page_number, section_title,
        section_type, verbatim_quote) for structured synthesis.
        """
        summaries: list[dict[str, Any]] = []
        for batch in results:
            category_value = batch.category
            if hasattr(category_value, "value"):
                category_value = category_value.value
            high_conf = [
                e for e in (batch.extractions or [])
                if e.confidence >= 0.7
            ]
            summaries.append({
                "category": category_value,
                "question_count": len(batch.extractions or []),
                "high_confidence_count": len(high_conf),
                "answers": [
                    {
                        "question_id": e.question_id,
                        "answer": e.answer,
                        "confidence": e.confidence,
                        "citations": [
                            {
                                "page_number": c.page_number,
                                "section_title": c.section_title,
                                "section_type": c.section_type,
                                "verbatim_quote": c.verbatim_quote,
                            }
                            for c in e.citations[:3]
                        ],
                    }
                    for e in high_conf
                ],
            })
        return summaries

    # ── Citation index builder ──────────────────────────────────────

    @staticmethod
    def _build_citation_index(
        results: list[BatchExtractionResult],
    ) -> dict[int, list[CitationRef]]:
        """Build a page-keyed index of ``CitationRef`` objects.

        Filters to confidence >= 0.5 and deduplicates by
        (page_number, first 50 chars of quote).
        """
        seen: set[tuple[int, str]] = set()
        index: dict[int, list[CitationRef]] = {}

        for batch in results:
            for extraction in (batch.extractions or []):
                if extraction.confidence < 0.5:
                    continue
                for c in extraction.citations:
                    quote_prefix = c.verbatim_quote[:50] if c.verbatim_quote else ""
                    dedup_key = (c.page_number, quote_prefix)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    ref = CitationRef(
                        page_number=c.page_number,
                        source_type=c.section_type,
                        section_title=c.section_title,
                        verbatim_quote=c.verbatim_quote,
                    )
                    index.setdefault(c.page_number, []).append(ref)

        return index

    # ── Legacy response parser ──────────────────────────────────────

    def _parse_summary(
        self,
        response: str,
        results: list[BatchExtractionResult],
        metadata: dict[str, Any],
    ) -> UnderwriterSummary:
        """Parse LLM response into an UnderwriterSummary."""
        parsed = self._client.extract_json(response)

        # Handle malformed LLM output: parsed may be a list, str, or empty
        if not isinstance(parsed, dict):
            log.warning("Synthesis response was not a dict (got %s), using fallback", type(parsed).__name__)
            parsed = {}

        total_answered = sum(
            len(batch.extractions or []) for batch in results
        )
        high_conf_count = sum(
            sum(1 for e in (batch.extractions or []) if e.confidence >= 0.7)
            for batch in results
        )

        raw_sections = parsed.get("sections", [])
        if not isinstance(raw_sections, list):
            raw_sections = []
        sections = [
            SynthesisSection(
                title=s.get("title", "") if isinstance(s, dict) else str(s),
                content=s.get("content", "") if isinstance(s, dict) else "",
                source_categories=s.get("source_categories", []) if isinstance(s, dict) else [],
                key_findings=s.get("key_findings", []) if isinstance(s, dict) else [],
            )
            for s in raw_sections
        ]

        raw_risk = parsed.get("risk_factors", [])
        if not isinstance(raw_risk, list):
            raw_risk = []

        return UnderwriterSummary(
            document_id=metadata.get("doc_id", ""),
            patient_demographics=str(parsed.get("patient_demographics", "") or ""),
            sections=sections,
            risk_factors=raw_risk,
            overall_assessment=str(parsed.get("overall_assessment", "") or ""),
            total_questions_answered=total_answered,
            high_confidence_count=high_conf_count,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    # ── Structured response parser ──────────────────────────────────

    def _parse_aps_summary(
        self,
        response: str,
        results: list[BatchExtractionResult],
        metadata: dict[str, Any],
        citation_index: dict[int, list[CitationRef]],
    ) -> APSSummary:
        """Parse LLM response into an APSSummary with typed sections."""
        parsed = self._client.extract_json(response)

        total_answered = sum(
            len(batch.extractions or []) for batch in results
        )
        high_conf_count = sum(
            sum(1 for e in (batch.extractions or []) if e.confidence >= 0.7)
            for batch in results
        )

        # Parse demographics
        demo_data = parsed.get("demographics", {})
        demographics = PatientDemographics(
            full_name=demo_data.get("full_name", ""),
            date_of_birth=demo_data.get("date_of_birth", ""),
            age=demo_data.get("age", ""),
            gender=demo_data.get("gender", ""),
            ssn_last4=demo_data.get("ssn_last4", ""),
            address=demo_data.get("address", ""),
            phone=demo_data.get("phone", ""),
            insurance_id=demo_data.get("insurance_id", ""),
            employer=demo_data.get("employer", ""),
            occupation=demo_data.get("occupation", ""),
        )
        # Fallback if structured demographics missing but flat string present
        if not demographics.full_name and parsed.get("patient_demographics"):
            demographics.raw_text = parsed["patient_demographics"]

        # Parse sections
        sections = [
            self._parse_aps_section(s)
            for s in parsed.get("sections", [])
        ]

        # Parse risk classification
        rc_data = parsed.get("risk_classification", {})
        risk_classification = RiskClassification(
            tier=rc_data.get("tier", ""),
            table_rating=rc_data.get("table_rating", ""),
            debit_credits=rc_data.get("debit_credits", ""),
            rationale=rc_data.get("rationale", ""),
        )

        # Parse red flags
        red_flags = [
            RedFlag(
                description=rf.get("description", ""),
                severity=rf.get("severity", "MODERATE"),
                category=rf.get("category", ""),
                citations=[
                    CitationRef(
                        page_number=c.get("page_number", 0),
                        date=c.get("date", ""),
                        source_type=c.get("source_type", ""),
                    )
                    for c in rf.get("citations", [])
                ],
            )
            for rf in parsed.get("red_flags", [])
        ]

        return APSSummary(
            document_id=metadata.get("doc_id", ""),
            demographics=demographics,
            sections=sections,
            risk_classification=risk_classification,
            risk_factors=parsed.get("risk_factors", []),
            red_flags=red_flags,
            overall_assessment=parsed.get("overall_assessment", ""),
            citation_index=citation_index,
            total_questions_answered=total_answered,
            high_confidence_count=high_conf_count,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _parse_citation_refs(citations_data: list[dict[str, Any]]) -> list[CitationRef]:
        """Parse a list of citation dicts into CitationRef objects."""
        return [
            CitationRef(
                page_number=c.get("page_number", 0),
                date=c.get("date", ""),
                source_type=c.get("source_type", ""),
                section_title=c.get("section_title", ""),
                verbatim_quote=c.get("verbatim_quote", ""),
            )
            for c in citations_data
        ]

    @staticmethod
    def _parse_aps_section(data: dict[str, Any]) -> APSSection:
        """Parse a single section dict into an APSSection."""
        parse_refs = SynthesisPipeline._parse_citation_refs

        findings = [
            Finding(
                text=f.get("text", ""),
                severity=f.get("severity", "INFORMATIONAL"),
                citations=parse_refs(f.get("citations", [])),
            )
            for f in data.get("findings", [])
        ]

        conditions = [
            Condition(
                name=c.get("name", ""),
                icd10_code=c.get("icd10_code", ""),
                onset_date=c.get("onset_date", ""),
                status=c.get("status", ""),
                severity=c.get("severity", ""),
                citations=parse_refs(c.get("citations", [])),
            )
            for c in data.get("conditions", [])
        ]

        medications = [
            Medication(
                name=m.get("name", ""),
                dose=m.get("dose", ""),
                frequency=m.get("frequency", ""),
                route=m.get("route", ""),
                prescriber=m.get("prescriber", ""),
                start_date=m.get("start_date", ""),
                citations=parse_refs(m.get("citations", [])),
            )
            for m in data.get("medications", [])
        ]

        lab_results = [
            LabResult(
                test_name=lr.get("test_name", ""),
                value=lr.get("value", ""),
                unit=lr.get("unit", ""),
                reference_range=lr.get("reference_range", ""),
                flag=lr.get("flag", ""),
                date=lr.get("date", ""),
                citations=parse_refs(lr.get("citations", [])),
            )
            for lr in data.get("lab_results", [])
        ]

        imaging_results = [
            ImagingResult(
                modality=ir.get("modality", ""),
                body_part=ir.get("body_part", ""),
                finding=ir.get("finding", ""),
                impression=ir.get("impression", ""),
                date=ir.get("date", ""),
                citations=parse_refs(ir.get("citations", [])),
            )
            for ir in data.get("imaging_results", [])
        ]

        encounters = [
            Encounter(
                date=enc.get("date", ""),
                provider=enc.get("provider", ""),
                encounter_type=enc.get("encounter_type", ""),
                summary=enc.get("summary", ""),
                citations=parse_refs(enc.get("citations", [])),
            )
            for enc in data.get("encounters", [])
        ]

        vital_signs = [
            VitalSign(
                name=vs.get("name", ""),
                value=vs.get("value", ""),
                date=vs.get("date", ""),
                flag=vs.get("flag", ""),
                citations=parse_refs(vs.get("citations", [])),
            )
            for vs in data.get("vital_signs", [])
        ]

        allergies = [
            Allergy(
                allergen=a.get("allergen", ""),
                reaction=a.get("reaction", ""),
                severity=a.get("severity", ""),
                citations=parse_refs(a.get("citations", [])),
            )
            for a in data.get("allergies", [])
        ]

        surgical_history = [
            SurgicalHistory(
                procedure=sh.get("procedure", ""),
                date=sh.get("date", ""),
                outcome=sh.get("outcome", ""),
                complications=sh.get("complications", ""),
                citations=parse_refs(sh.get("citations", [])),
            )
            for sh in data.get("surgical_history", [])
        ]

        return APSSection(
            section_key=data.get("section_key", ""),
            section_number=data.get("section_number", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            source_categories=data.get("source_categories", []),
            findings=findings,
            conditions=conditions,
            medications=medications,
            lab_results=lab_results,
            imaging_results=imaging_results,
            encounters=encounters,
            vital_signs=vital_signs,
            allergies=allergies,
            surgical_history=surgical_history,
        )

    # ══════════════════════════════════════════════════════════════════
    # Underwriting template synthesis
    # ══════════════════════════════════════════════════════════════════

    # Maps question_id → (condition_name, time_qualifier)
    _YN_CONDITION_MAP: dict[str, tuple[str, str]] = {
        "uw-sub-001": ("Alcohol Treatment", "within 2 years"),
        "uw-sub-002": ("Tobacco Use", ""),
        "uw-sub-003": ("Drug Use/Treatment", "within 3 years"),
        "uw-crit-001": ("Dementia", ""),
        "uw-crit-002": ("CVA/Stroke", "within 1 year"),
        "uw-crit-003": ("Myocardial Infarction", "within 3 months"),
        "uw-crit-004": ("Renal Dialysis", ""),
        "uw-crit-005": ("Cancer", ""),
        "uw-crit-006": ("Cardiac Valve Replacement", ""),
        "uw-crit-007": ("AIDS/HIV", ""),
        "uw-mental-001": ("Disability (Mental Disorder)", ""),
        "uw-mental-002": ("Suicide Attempt", "within 1 year"),
        "uw-mental-003": ("Psychiatric Hospitalization", ""),
        "uw-other-001": ("Cirrhosis", ""),
        "uw-other-002": ("Gastric Bypass", "within 6 months"),
        "uw-res-001": ("Foreign Residence/Travel", ""),
        "uw-res-002": ("Travel-Related Concerns", ""),
    }

    _YES_PATTERNS = re.compile(r"^(y(es)?|positive|confirmed|true)\b", re.IGNORECASE)
    _NO_PATTERNS = re.compile(
        r"^(n(o)?|negative|denied|not\s+found|none|false)\b", re.IGNORECASE,
    )

    def synthesize_underwriting(
        self,
        extraction_results: list[BatchExtractionResult],
        document_metadata: dict[str, Any] | None = None,
    ) -> UnderwritingAPSSummary:
        """Build an ``UnderwritingAPSSummary`` from extraction results.

        This is a **deterministic** post-processing step — no LLM call.
        It restructures the extraction answers into the underwriting
        template format with Y/N conditions and narrative sections.
        """
        metadata = document_metadata or {}

        # Build a flat question_id → extraction lookup
        answer_map: dict[str, Any] = {}
        for batch in extraction_results:
            for ext in (batch.extractions or []):
                answer_map[ext.question_id] = ext

        # Demographics
        policy_number = self._get_answer_text(answer_map, "uw-demo-004")
        aps_date_range = self._get_answer_text(answer_map, "uw-demo-005")
        total_pages = metadata.get("total_pages", 0)

        # Y/N conditions
        yn_conditions: list[YNCondition] = []
        for qid, (condition_name, time_qual) in self._YN_CONDITION_MAP.items():
            ext = answer_map.get(qid)
            if ext is None:
                yn_conditions.append(
                    YNCondition(condition_name=condition_name, time_qualifier=time_qual, answer_yn="Unknown"),
                )
                continue
            answer_yn = self._parse_yn(ext.answer)
            citations = self._convert_citations(ext.citations)
            yn_conditions.append(
                YNCondition(
                    condition_name=condition_name,
                    time_qualifier=time_qual,
                    answer_yn=answer_yn,
                    detail=ext.answer,
                    citations=citations,
                ),
            )

        # Narrative sections
        morbidity_concerns = self._get_answer_text(answer_map, "uw-morb-001")
        morb2 = self._get_answer_text(answer_map, "uw-morb-002")
        if morb2 and morb2 != "Not found":
            morbidity_concerns = f"{morbidity_concerns}\n\n{morb2}" if morbidity_concerns else morb2
        morbidity_citations = self._collect_citations(answer_map, ["uw-morb-001", "uw-morb-002"])

        mortality_concerns = self._get_answer_text(answer_map, "uw-mort-001")
        mort2 = self._get_answer_text(answer_map, "uw-mort-002")
        if mort2 and mort2 != "Not found":
            mortality_concerns = f"{mortality_concerns}\n\n{mort2}" if mortality_concerns else mort2
        mortality_citations = self._collect_citations(answer_map, ["uw-mort-001", "uw-mort-002"])

        residence_travel = self._get_answer_text(answer_map, "uw-res-001")
        res2 = self._get_answer_text(answer_map, "uw-res-002")
        if res2 and res2 != "Not found":
            residence_travel = f"{residence_travel}\n\n{res2}" if residence_travel else res2
        residence_citations = self._collect_citations(answer_map, ["uw-res-001", "uw-res-002"])

        # Build the base APSSummary fields
        demographics = PatientDemographics(
            full_name=self._get_answer_text(answer_map, "uw-demo-001"),
            date_of_birth=self._get_answer_text(answer_map, "uw-demo-002"),
            gender=self._get_answer_text(answer_map, "uw-demo-003"),
        )

        total_answered = sum(len(batch.extractions or []) for batch in extraction_results)
        high_conf_count = sum(
            sum(1 for e in (batch.extractions or []) if e.confidence >= 0.7)
            for batch in extraction_results
        )

        citation_index = self._build_citation_index(extraction_results)

        return UnderwritingAPSSummary(
            document_id=metadata.get("doc_id", ""),
            demographics=demographics,
            citation_index=citation_index,
            total_questions_answered=total_answered,
            high_confidence_count=high_conf_count,
            generated_at=datetime.now(timezone.utc).isoformat(),
            policy_number=policy_number,
            aps_date_range=aps_date_range,
            total_document_pages=total_pages,
            yn_conditions=yn_conditions,
            morbidity_concerns=morbidity_concerns,
            morbidity_citations=morbidity_citations,
            mortality_concerns=mortality_concerns,
            mortality_citations=mortality_citations,
            residence_travel=residence_travel,
            residence_citations=residence_citations,
        )

    # ── Underwriting helpers ────────────────────────────────────────

    @staticmethod
    def _get_answer_text(answer_map: dict[str, Any], question_id: str) -> str:
        ext = answer_map.get(question_id)
        if ext is None:
            return ""
        answer = ext.answer if hasattr(ext, "answer") else ""
        return answer if answer != "Not found" else ""

    @classmethod
    def _parse_yn(cls, answer: str) -> str:
        """Parse Y/N from free-text answer."""
        answer = answer.strip()
        if cls._YES_PATTERNS.search(answer):
            return "Y"
        if cls._NO_PATTERNS.search(answer):
            return "N"
        return "Unknown"

    @staticmethod
    def _convert_citations(citations: list[Any]) -> list[CitationRef]:
        """Convert ``Citation`` (models.py) to ``CitationRef`` (aps/models.py)."""
        refs: list[CitationRef] = []
        for c in citations[:3]:
            page = c.page_number if hasattr(c, "page_number") else 0
            section_title = c.section_title if hasattr(c, "section_title") else ""
            source_type = c.section_type if hasattr(c, "section_type") else ""
            quote = c.verbatim_quote if hasattr(c, "verbatim_quote") else ""
            refs.append(CitationRef(
                page_number=page,
                section_title=section_title,
                source_type=source_type,
                verbatim_quote=quote,
            ))
        return refs

    @classmethod
    def _collect_citations(
        cls, answer_map: dict[str, Any], question_ids: list[str],
    ) -> list[CitationRef]:
        """Collect and deduplicate citations from multiple questions."""
        all_refs: list[CitationRef] = []
        seen: set[int] = set()
        for qid in question_ids:
            ext = answer_map.get(qid)
            if ext is None:
                continue
            for ref in cls._convert_citations(ext.citations):
                if ref.page_number not in seen:
                    seen.add(ref.page_number)
                    all_refs.append(ref)
        return all_refs
