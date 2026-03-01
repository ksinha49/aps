"""Tests for evidence grounding checks."""

from __future__ import annotations

from scout_ai.synthesis.models import APSSection, APSSummary, CitationRef, Finding
from scout_ai.validation.checks.evidence_grounding import check_evidence_grounding
from scout_ai.validation.models import IssueSeverity, Rule, RuleCategory, RuleTarget


def _make_rule(rule_id: str, **params: object) -> Rule:
    return Rule(
        rule_id=rule_id,
        name=f"Rule {rule_id}",
        description="test",
        category=RuleCategory.EVIDENCE_GROUNDING,
        target=RuleTarget.FINDING,
        severity=IssueSeverity.ERROR,
        params=dict(params),
    )


class TestCitationsRequired:
    def _rules(self) -> list[Rule]:
        return [_make_rule("EG-001", require_citation_for=["CRITICAL", "SIGNIFICANT"])]

    def test_critical_without_citation(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    findings=[Finding(text="Critical cardiac event", severity="CRITICAL")],
                )
            ],
        )
        issues = check_evidence_grounding(summary, self._rules())
        assert len(issues) == 1
        assert issues[0].rule_id == "EG-001"

    def test_critical_with_citation(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    findings=[
                        Finding(
                            text="Critical cardiac event",
                            severity="CRITICAL",
                            citations=[CitationRef(page_number=5)],
                        )
                    ],
                )
            ],
        )
        issues = check_evidence_grounding(summary, self._rules())
        assert len(issues) == 0

    def test_minor_without_citation_ok(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    findings=[Finding(text="Normal checkup", severity="MINOR")],
                )
            ],
        )
        issues = check_evidence_grounding(summary, self._rules())
        assert len(issues) == 0


class TestCitationPageRange:
    def _rules(self) -> list[Rule]:
        return [_make_rule("EG-002")]

    def test_valid_page(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    findings=[
                        Finding(
                            text="Finding",
                            severity="MODERATE",
                            citations=[CitationRef(page_number=10)],
                        )
                    ],
                )
            ],
        )
        issues = check_evidence_grounding(summary, self._rules(), total_pages=50)
        assert len(issues) == 0

    def test_page_out_of_range(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    findings=[
                        Finding(
                            text="Finding",
                            severity="MODERATE",
                            citations=[CitationRef(page_number=999)],
                        )
                    ],
                )
            ],
        )
        issues = check_evidence_grounding(summary, self._rules(), total_pages=50)
        assert len(issues) == 1
        assert issues[0].rule_id == "EG-002"

    def test_zero_total_pages_skips(self) -> None:
        """EG-002 is skipped when total_pages=0 (unknown document length)."""
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    findings=[
                        Finding(
                            text="Finding",
                            severity="MODERATE",
                            citations=[CitationRef(page_number=999)],
                        )
                    ],
                )
            ],
        )
        issues = check_evidence_grounding(summary, self._rules(), total_pages=0)
        assert len(issues) == 0


class TestUncitedSeverityCap:
    def _rules(self) -> list[Rule]:
        return [_make_rule("EG-003", max_uncited_severity="MINOR")]

    def test_uncited_moderate_flagged(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    findings=[Finding(text="Uncited finding", severity="MODERATE")],
                )
            ],
        )
        issues = check_evidence_grounding(summary, self._rules())
        assert len(issues) == 1
        assert issues[0].rule_id == "EG-003"

    def test_uncited_minor_ok(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    findings=[Finding(text="Minor finding", severity="MINOR")],
                )
            ],
        )
        issues = check_evidence_grounding(summary, self._rules())
        assert len(issues) == 0

    def test_cited_moderate_ok(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    findings=[
                        Finding(
                            text="Cited finding",
                            severity="MODERATE",
                            citations=[CitationRef(page_number=3)],
                        )
                    ],
                )
            ],
        )
        issues = check_evidence_grounding(summary, self._rules())
        assert len(issues) == 0
