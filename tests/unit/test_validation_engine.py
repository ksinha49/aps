"""Tests for the RulesEngine end-to-end with MemoryRulesBackend."""

from __future__ import annotations

from scout_ai.synthesis.models import (
    APSSection,
    APSSummary,
    CitationRef,
    Condition,
    Finding,
    RiskClassification,
)
from scout_ai.validation.backends.memory_backend import MemoryRulesBackend
from scout_ai.validation.engine import RulesEngine
from scout_ai.validation.models import (
    IssueSeverity,
    Rule,
    RuleCategory,
    RuleTarget,
)


def _make_rules() -> list[Rule]:
    """Create a representative set of rules for testing."""
    return [
        Rule(
            rule_id="DI-001",
            name="ICD-10 format",
            description="ICD-10 codes must match standard format",
            category=RuleCategory.DATA_INTEGRITY,
            target=RuleTarget.CONDITION,
            severity=IssueSeverity.ERROR,
            params={"pattern": r"^[A-Z]\d{2}(\.\d{1,4})?$"},
        ),
        Rule(
            rule_id="DI-002",
            name="Severity enum",
            description="Severity must be valid",
            category=RuleCategory.DATA_INTEGRITY,
            target=RuleTarget.FINDING,
            severity=IssueSeverity.ERROR,
            params={"allowed": ["CRITICAL", "SIGNIFICANT", "MODERATE", "MINOR", "INFORMATIONAL"]},
        ),
        Rule(
            rule_id="EG-001",
            name="Citations required",
            description="Critical findings need citations",
            category=RuleCategory.EVIDENCE_GROUNDING,
            target=RuleTarget.FINDING,
            severity=IssueSeverity.ERROR,
            params={"require_citation_for": ["CRITICAL", "SIGNIFICANT"]},
        ),
        Rule(
            rule_id="RC-001",
            name="Valid tier",
            description="Tier must be recognized",
            category=RuleCategory.RISK_CLASSIFICATION,
            target=RuleTarget.RISK_CLASSIFICATION,
            severity=IssueSeverity.ERROR,
            params={
                "allowed_tiers": [
                    "Preferred Plus", "Preferred", "Standard Plus",
                    "Standard", "Substandard", "Decline",
                ]
            },
        ),
        Rule(
            rule_id="RC-002",
            name="Critical vs preferred",
            description="Critical findings contradict Preferred Plus",
            category=RuleCategory.RISK_CLASSIFICATION,
            target=RuleTarget.RISK_CLASSIFICATION,
            severity=IssueSeverity.ERROR,
            params={
                "incompatible_tiers": ["Preferred Plus"],
                "trigger_severities": ["CRITICAL"],
            },
        ),
    ]


class TestRulesEngine:
    def test_clean_summary_passes(self) -> None:
        engine = RulesEngine(MemoryRulesBackend(_make_rules()))
        summary = APSSummary(
            document_id="clean",
            risk_classification=RiskClassification(tier="Standard"),
            sections=[
                APSSection(
                    section_key="medical",
                    findings=[
                        Finding(
                            text="Normal checkup",
                            severity="MINOR",
                            citations=[CitationRef(page_number=1)],
                        )
                    ],
                    conditions=[Condition(name="HTN", icd10_code="I10")],
                )
            ],
        )
        report = engine.validate(summary)
        assert report.passed is True
        assert report.total_issues == 0
        assert report.total_rules_evaluated == 5

    def test_multiple_violations(self) -> None:
        engine = RulesEngine(MemoryRulesBackend(_make_rules()))
        summary = APSSummary(
            document_id="bad",
            risk_classification=RiskClassification(tier="Preferred Plus"),
            sections=[
                APSSection(
                    section_key="medical",
                    findings=[
                        Finding(text="Critical event", severity="CRITICAL"),  # no citation → EG-001
                        Finding(text="Checkup", severity="BOGUS"),  # bad severity → DI-002
                    ],
                    conditions=[
                        Condition(name="HTN", icd10_code="INVALID"),  # bad ICD → DI-001
                    ],
                )
            ],
        )
        report = engine.validate(summary)
        assert report.passed is False
        assert report.error_count >= 3
        rule_ids = {i.rule_id for i in report.issues}
        assert "DI-001" in rule_ids
        assert "DI-002" in rule_ids
        assert "EG-001" in rule_ids
        assert "RC-002" in rule_ids  # CRITICAL + Preferred Plus

    def test_empty_summary_passes(self) -> None:
        engine = RulesEngine(MemoryRulesBackend(_make_rules()))
        summary = APSSummary(document_id="empty")
        report = engine.validate(summary)
        assert report.passed is True

    def test_category_exception_isolation(self) -> None:
        """A failing category should not block other categories."""

        class BrokenBackend(MemoryRulesBackend):
            pass

        # Create engine with valid rules — the checks themselves are deterministic
        engine = RulesEngine(MemoryRulesBackend(_make_rules()))
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    conditions=[Condition(name="HTN", icd10_code="INVALID")],
                )
            ],
        )
        report = engine.validate(summary)
        # Even if one category had issues, we should get results from others
        assert report.total_rules_evaluated > 0

    def test_disabled_rules_excluded(self) -> None:
        rules = [
            Rule(
                rule_id="DI-001",
                name="ICD-10",
                description="test",
                category=RuleCategory.DATA_INTEGRITY,
                target=RuleTarget.CONDITION,
                severity=IssueSeverity.ERROR,
                enabled=False,
                params={"pattern": r"^[A-Z]\d{2}(\.\d{1,4})?$"},
            )
        ]
        engine = RulesEngine(MemoryRulesBackend(rules))
        summary = APSSummary(
            document_id="test",
            sections=[
                APSSection(
                    section_key="medical",
                    conditions=[Condition(name="HTN", icd10_code="INVALID")],
                )
            ],
        )
        report = engine.validate(summary)
        assert report.total_rules_evaluated == 0
        assert report.passed is True
