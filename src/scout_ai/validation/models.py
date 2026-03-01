"""Validation data models: rules, issues, and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class IssueSeverity(str, Enum):
    """Severity of a validation issue."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class RuleCategory(str, Enum):
    """Category of a validation rule.

    Domains may define additional categories beyond these core values.
    Use string values directly for domain-specific categories.
    """

    DATA_INTEGRITY = "data_integrity"
    MEDICAL_BUSINESS = "medical_business"
    EVIDENCE_GROUNDING = "evidence_grounding"
    RISK_CLASSIFICATION = "risk_classification"

    @classmethod
    def _missing_(cls, value: object) -> RuleCategory | None:
        """Allow arbitrary string values for domain extensibility."""
        if isinstance(value, str):
            obj = str.__new__(cls, value)
            obj._value_ = value
            obj._name_ = value.upper()
            return obj
        return None


class RuleTarget(str, Enum):
    """What entity type a rule validates.

    Domains may define additional targets beyond these core values.
    Use string values directly for domain-specific targets.
    """

    FINDING = "finding"
    CONDITION = "condition"
    MEDICATION = "medication"
    LAB_RESULT = "lab_result"
    RED_FLAG = "red_flag"
    RISK_CLASSIFICATION = "risk_classification"
    SECTION = "section"
    SUMMARY = "summary"

    @classmethod
    def _missing_(cls, value: object) -> RuleTarget | None:
        """Allow arbitrary string values for domain extensibility."""
        if isinstance(value, str):
            obj = str.__new__(cls, value)
            obj._value_ = value
            obj._name_ = value.upper()
            return obj
        return None


@dataclass(frozen=True)
class Rule:
    """A single validation rule loaded from a backend."""

    rule_id: str
    name: str
    description: str
    category: RuleCategory
    target: RuleTarget
    severity: IssueSeverity = IssueSeverity.WARNING
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    version: int = 1


@dataclass
class ValidationIssue:
    """A single issue found during validation."""

    rule_id: str
    rule_name: str
    severity: IssueSeverity
    category: RuleCategory
    message: str
    section_key: str = ""
    field_path: str = ""
    entity_name: str = ""
    actual_value: str = ""
    expected_hint: str = ""


@dataclass
class ValidationReport:
    """Aggregated result of running all rules against a summary."""

    document_id: str
    total_rules_evaluated: int = 0
    total_issues: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    passed: bool = True
    validated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    rules_version: int = 1

    def has_errors(self) -> bool:
        """Return True if any ERROR-level issues exist."""
        return self.error_count > 0

    def issues_by_category(self) -> dict[RuleCategory, list[ValidationIssue]]:
        """Group issues by their rule category."""
        grouped: dict[RuleCategory, list[ValidationIssue]] = {}
        for issue in self.issues:
            grouped.setdefault(issue.category, []).append(issue)
        return grouped
