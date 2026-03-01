"""Tests for validation data models: Rule, ValidationIssue, ValidationReport."""

from __future__ import annotations

from scout_ai.validation.models import (
    IssueSeverity,
    Rule,
    RuleCategory,
    RuleTarget,
    ValidationIssue,
    ValidationReport,
)


class TestRule:
    def test_defaults(self) -> None:
        rule = Rule(
            rule_id="TEST-001",
            name="Test rule",
            description="A test",
            category=RuleCategory.DATA_INTEGRITY,
            target=RuleTarget.FINDING,
        )
        assert rule.severity == IssueSeverity.WARNING
        assert rule.enabled is True
        assert rule.params == {}
        assert rule.version == 1

    def test_frozen(self) -> None:
        rule = Rule(
            rule_id="X",
            name="X",
            description="X",
            category=RuleCategory.DATA_INTEGRITY,
            target=RuleTarget.FINDING,
        )
        try:
            rule.rule_id = "Y"  # type: ignore[misc]
            assert False, "Should not allow mutation"
        except AttributeError:
            pass


class TestValidationIssue:
    def test_creation(self) -> None:
        issue = ValidationIssue(
            rule_id="DI-001",
            rule_name="ICD-10 format",
            severity=IssueSeverity.ERROR,
            category=RuleCategory.DATA_INTEGRITY,
            message="Invalid ICD-10",
        )
        assert issue.rule_id == "DI-001"
        assert issue.section_key == ""
        assert issue.actual_value == ""


class TestValidationReport:
    def test_defaults(self) -> None:
        report = ValidationReport(document_id="doc1")
        assert report.passed is True
        assert report.total_issues == 0
        assert report.error_count == 0
        assert report.issues == []
        assert report.validated_at != ""

    def test_has_errors_false_when_empty(self) -> None:
        report = ValidationReport(document_id="doc1")
        assert report.has_errors() is False

    def test_has_errors_true(self) -> None:
        report = ValidationReport(document_id="doc1", error_count=2)
        assert report.has_errors() is True

    def test_issues_by_category(self) -> None:
        issues = [
            ValidationIssue(
                rule_id="DI-001",
                rule_name="ICD",
                severity=IssueSeverity.ERROR,
                category=RuleCategory.DATA_INTEGRITY,
                message="bad code",
            ),
            ValidationIssue(
                rule_id="EG-001",
                rule_name="Citations",
                severity=IssueSeverity.ERROR,
                category=RuleCategory.EVIDENCE_GROUNDING,
                message="no citation",
            ),
            ValidationIssue(
                rule_id="DI-002",
                rule_name="Severity",
                severity=IssueSeverity.ERROR,
                category=RuleCategory.DATA_INTEGRITY,
                message="bad severity",
            ),
        ]
        report = ValidationReport(document_id="doc1", issues=issues)
        grouped = report.issues_by_category()
        assert len(grouped[RuleCategory.DATA_INTEGRITY]) == 2
        assert len(grouped[RuleCategory.EVIDENCE_GROUNDING]) == 1
