"""Tests for risk classification checks."""

from __future__ import annotations

from scout_ai.synthesis.models import APSSection, APSSummary, Finding, RiskClassification
from scout_ai.validation.checks.risk_classification import check_risk_classification
from scout_ai.validation.models import IssueSeverity, Rule, RuleCategory, RuleTarget


def _make_rule(rule_id: str, **params: object) -> Rule:
    return Rule(
        rule_id=rule_id,
        name=f"Rule {rule_id}",
        description="test",
        category=RuleCategory.RISK_CLASSIFICATION,
        target=RuleTarget.RISK_CLASSIFICATION,
        severity=IssueSeverity.ERROR,
        params=dict(params),
    )


class TestValidTier:
    def _rules(self) -> list[Rule]:
        return [
            _make_rule(
                "RC-001",
                allowed_tiers=[
                    "Preferred Plus",
                    "Preferred",
                    "Standard Plus",
                    "Standard",
                    "Substandard",
                    "Decline",
                ],
            )
        ]

    def test_valid_tier(self) -> None:
        summary = APSSummary(
            document_id="test",
            risk_classification=RiskClassification(tier="Standard"),
        )
        issues = check_risk_classification(summary, self._rules())
        assert len(issues) == 0

    def test_invalid_tier(self) -> None:
        summary = APSSummary(
            document_id="test",
            risk_classification=RiskClassification(tier="Super Premium"),
        )
        issues = check_risk_classification(summary, self._rules())
        assert len(issues) == 1
        assert issues[0].rule_id == "RC-001"

    def test_empty_tier_ok(self) -> None:
        summary = APSSummary(
            document_id="test",
            risk_classification=RiskClassification(tier=""),
        )
        issues = check_risk_classification(summary, self._rules())
        assert len(issues) == 0


class TestCriticalVsTier:
    def _rules(self) -> list[Rule]:
        return [
            _make_rule(
                "RC-002",
                incompatible_tiers=["Preferred Plus"],
                trigger_severities=["CRITICAL"],
            )
        ]

    def test_critical_with_preferred_plus(self) -> None:
        summary = APSSummary(
            document_id="test",
            risk_classification=RiskClassification(tier="Preferred Plus"),
            sections=[
                APSSection(
                    section_key="medical",
                    findings=[Finding(text="Critical cardiac event", severity="CRITICAL")],
                )
            ],
        )
        issues = check_risk_classification(summary, self._rules())
        assert len(issues) == 1
        assert issues[0].rule_id == "RC-002"

    def test_critical_with_standard(self) -> None:
        summary = APSSummary(
            document_id="test",
            risk_classification=RiskClassification(tier="Standard"),
            sections=[
                APSSection(
                    section_key="medical",
                    findings=[Finding(text="Critical cardiac event", severity="CRITICAL")],
                )
            ],
        )
        issues = check_risk_classification(summary, self._rules())
        assert len(issues) == 0

    def test_no_critical_with_preferred_plus(self) -> None:
        summary = APSSummary(
            document_id="test",
            risk_classification=RiskClassification(tier="Preferred Plus"),
            sections=[
                APSSection(
                    section_key="medical",
                    findings=[Finding(text="Normal finding", severity="MINOR")],
                )
            ],
        )
        issues = check_risk_classification(summary, self._rules())
        assert len(issues) == 0
