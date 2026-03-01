"""Tests for medical business rule checks."""

from __future__ import annotations

from scout_ai.synthesis.models import (
    APSSection,
    APSSummary,
    Condition,
    Finding,
    LabResult,
    Medication,
    RedFlag,
    VitalSign,
)
from scout_ai.validation.checks.medical_business import check_medical_business
from scout_ai.validation.models import IssueSeverity, Rule, RuleCategory, RuleTarget


def _make_rule(rule_id: str, **params: object) -> Rule:
    return Rule(
        rule_id=rule_id,
        name=f"Rule {rule_id}",
        description="test",
        category=RuleCategory.MEDICAL_BUSINESS,
        target=RuleTarget.LAB_RESULT,
        severity=IssueSeverity.WARNING,
        params=dict(params),
    )


class TestHbA1cSeverity:
    def _rules(self) -> list[Rule]:
        return [
            _make_rule(
                "MB-001",
                test_names=["HbA1c", "A1C"],
                threshold=7.0,
                min_severity="MODERATE",
            )
        ]

    def test_high_hba1c_low_severity(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="labs",
                    lab_results=[LabResult(test_name="HbA1c", value="8.2 %")],
                    findings=[Finding(text="Elevated A1C", severity="MINOR")],
                )
            ],
        )
        issues = check_medical_business(summary, self._rules())
        assert len(issues) == 1
        assert issues[0].rule_id == "MB-001"

    def test_high_hba1c_adequate_severity(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="labs",
                    lab_results=[LabResult(test_name="HbA1c", value="8.2 %")],
                    findings=[Finding(text="Elevated A1C", severity="SIGNIFICANT")],
                )
            ],
        )
        issues = check_medical_business(summary, self._rules())
        assert len(issues) == 0

    def test_normal_hba1c_no_issue(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="labs",
                    lab_results=[LabResult(test_name="HbA1c", value="5.7 %")],
                    findings=[],
                )
            ],
        )
        issues = check_medical_business(summary, self._rules())
        assert len(issues) == 0


class TestBMIRiskFactor:
    def _rules(self) -> list[Rule]:
        return [
            _make_rule(
                "MB-002",
                vital_names=["BMI", "Body Mass Index"],
                threshold=30.0,
            )
        ]

    def test_high_bmi_missing_risk(self) -> None:
        summary = APSSummary(
            document_id="test",
            risk_factors=["Diabetes"],
            sections=[
                APSSection(
                    section_key="vitals",
                    vital_signs=[VitalSign(name="BMI", value="35.2")],
                )
            ],
        )
        issues = check_medical_business(summary, self._rules())
        assert len(issues) == 1
        assert issues[0].rule_id == "MB-002"

    def test_high_bmi_in_risk_factors(self) -> None:
        summary = APSSummary(
            document_id="test",
            risk_factors=["Obesity (BMI 35.2)"],
            sections=[
                APSSection(
                    section_key="vitals",
                    vital_signs=[VitalSign(name="BMI", value="35.2")],
                )
            ],
        )
        issues = check_medical_business(summary, self._rules())
        assert len(issues) == 0


class TestAutoCriticalConditions:
    def _rules(self) -> list[Rule]:
        return [
            _make_rule(
                "MB-003",
                condition_patterns=[
                    r"(?i)\bcancer\b",
                    r"(?i)\bHIV\b",
                    r"(?i)\bleukemia\b",
                ],
            )
        ]

    def test_cancer_not_critical(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    conditions=[Condition(name="Breast Cancer", severity="SIGNIFICANT")],
                )
            ],
        )
        issues = check_medical_business(summary, self._rules())
        assert len(issues) == 1
        assert "Breast Cancer" in issues[0].message

    def test_cancer_already_critical(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    conditions=[Condition(name="Breast Cancer", severity="CRITICAL")],
                )
            ],
        )
        issues = check_medical_business(summary, self._rules())
        assert len(issues) == 0

    def test_non_matching_condition(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    conditions=[Condition(name="Hypertension", severity="MODERATE")],
                )
            ],
        )
        issues = check_medical_business(summary, self._rules())
        assert len(issues) == 0


class TestControlledSubstances:
    def _rules(self) -> list[Rule]:
        return [
            _make_rule(
                "MB-004",
                controlled_patterns=[
                    r"(?i)\boxycodone\b",
                    r"(?i)\balprazolam\b",
                    r"(?i)\bfentanyl\b",
                ],
                min_concurrent=2,
            )
        ]

    def test_multiple_controlled_no_red_flag(self) -> None:
        summary = APSSummary(
            document_id="test",
            red_flags=[],
            sections=[
                APSSection(
                    section_key="meds",
                    medications=[
                        Medication(name="Oxycodone 10mg"),
                        Medication(name="Alprazolam 0.5mg"),
                    ],
                )
            ],
        )
        issues = check_medical_business(summary, self._rules())
        assert len(issues) == 1
        assert issues[0].rule_id == "MB-004"

    def test_multiple_controlled_with_red_flag(self) -> None:
        summary = APSSummary(
            document_id="test",
            red_flags=[RedFlag(description="Multiple controlled substances")],
            sections=[
                APSSection(
                    section_key="meds",
                    medications=[
                        Medication(name="Oxycodone 10mg"),
                        Medication(name="Alprazolam 0.5mg"),
                    ],
                )
            ],
        )
        issues = check_medical_business(summary, self._rules())
        assert len(issues) == 0

    def test_single_controlled_no_issue(self) -> None:
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="meds",
                    medications=[Medication(name="Oxycodone 10mg")],
                )
            ],
        )
        issues = check_medical_business(summary, self._rules())
        assert len(issues) == 0
