"""Tests for data integrity validation checks."""

from __future__ import annotations

from scout_ai.synthesis.models import (
    APSSection,
    APSSummary,
    Condition,
    Finding,
    LabResult,
    Medication,
)
from scout_ai.validation.checks.data_integrity import check_data_integrity
from scout_ai.validation.models import IssueSeverity, Rule, RuleCategory, RuleTarget


def _make_rule(rule_id: str, **params: object) -> Rule:
    return Rule(
        rule_id=rule_id,
        name=f"Rule {rule_id}",
        description="test",
        category=RuleCategory.DATA_INTEGRITY,
        target=RuleTarget.CONDITION,
        severity=IssueSeverity.ERROR,
        params=dict(params),
    )


class TestICD10Codes:
    def _rules(self) -> list[Rule]:
        return [_make_rule("DI-001", pattern=r"^[A-Z]\d{2}(\.\d{1,4})?$")]

    def test_valid_codes(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    conditions=[
                        Condition(name="HTN", icd10_code="I10"),
                        Condition(name="Diabetes", icd10_code="E11.65"),
                    ],
                )
            ],
        )
        issues = check_data_integrity(summary, self._rules())
        assert len(issues) == 0

    def test_invalid_code(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    conditions=[Condition(name="HTN", icd10_code="INVALID")],
                )
            ],
        )
        issues = check_data_integrity(summary, self._rules())
        assert len(issues) == 1
        assert issues[0].rule_id == "DI-001"
        assert "INVALID" in issues[0].message

    def test_empty_code_skipped(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    conditions=[Condition(name="HTN", icd10_code="")],
                )
            ],
        )
        issues = check_data_integrity(summary, self._rules())
        assert len(issues) == 0


class TestSeverityEnum:
    def _rules(self) -> list[Rule]:
        return [
            _make_rule(
                "DI-002",
                allowed=["CRITICAL", "SIGNIFICANT", "MODERATE", "MINOR", "INFORMATIONAL"],
            )
        ]

    def test_valid_severity(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    findings=[Finding(text="Normal", severity="MINOR")],
                )
            ],
        )
        issues = check_data_integrity(summary, self._rules())
        assert len(issues) == 0

    def test_invalid_severity(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    findings=[Finding(text="Bad", severity="BOGUS")],
                )
            ],
        )
        issues = check_data_integrity(summary, self._rules())
        assert len(issues) == 1
        assert issues[0].rule_id == "DI-002"


class TestLabFlags:
    def _rules(self) -> list[Rule]:
        return [_make_rule("DI-003", allowed=["H", "L", "C", ""])]

    def test_valid_flags(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="labs",
                    lab_results=[
                        LabResult(test_name="Glucose", flag="H"),
                        LabResult(test_name="WBC", flag=""),
                    ],
                )
            ],
        )
        issues = check_data_integrity(summary, self._rules())
        assert len(issues) == 0

    def test_invalid_flag(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="labs",
                    lab_results=[LabResult(test_name="Glucose", flag="X")],
                )
            ],
        )
        issues = check_data_integrity(summary, self._rules())
        assert len(issues) == 1


class TestDateFormats:
    def _rules(self) -> list[Rule]:
        return [
            _make_rule(
                "DI-004",
                patterns=[
                    r"^\d{2}/\d{2}/\d{4}$",
                    r"^\d{2}/\d{4}$",
                    r"^\d{4}-\d{2}-\d{2}$",
                    r"^\d{4}$",
                ],
            )
        ]

    def test_valid_dates(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    conditions=[Condition(name="HTN", onset_date="01/15/2023")],
                    medications=[Medication(name="Metformin", start_date="2023-01-15")],
                )
            ],
        )
        issues = check_data_integrity(summary, self._rules())
        assert len(issues) == 0

    def test_invalid_date(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    conditions=[Condition(name="HTN", onset_date="January 15th")],
                )
            ],
        )
        issues = check_data_integrity(summary, self._rules())
        assert len(issues) == 1
        assert issues[0].rule_id == "DI-004"
