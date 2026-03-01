"""Rules engine: loads rules from a backend and dispatches to check modules."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scout_ai.validation.checks.data_integrity import check_data_integrity
from scout_ai.validation.checks.evidence_grounding import check_evidence_grounding
from scout_ai.validation.checks.medical_business import check_medical_business
from scout_ai.validation.checks.risk_classification import check_risk_classification
from scout_ai.validation.models import (
    IssueSeverity,
    RuleCategory,
    ValidationReport,
)

if TYPE_CHECKING:
    from scout_ai.synthesis.models import APSSummary
    from scout_ai.validation.backends.protocol import IRulesBackend

log = logging.getLogger(__name__)


class RulesEngine:
    """Validates an APSSummary against business rules loaded from a backend.

    Validation is pure computation — no LLM calls.  Each category's check
    module runs independently; a failure in one category does not block others.
    """

    def __init__(self, backend: IRulesBackend) -> None:
        self._backend = backend

    def validate(
        self,
        summary: APSSummary,
        *,
        total_pages: int = 0,
    ) -> ValidationReport:
        """Run all enabled rules against the summary and return a report."""
        all_rules = self._backend.list_rules(enabled_only=True)
        rules_version = self._backend.get_version()

        report = ValidationReport(
            document_id=summary.document_id,
            total_rules_evaluated=len(all_rules),
            rules_version=rules_version,
        )

        # Dispatch to each category's check module
        dispatchers = [
            (RuleCategory.DATA_INTEGRITY, self._run_data_integrity),
            (RuleCategory.MEDICAL_BUSINESS, self._run_medical_business),
            (RuleCategory.EVIDENCE_GROUNDING, self._run_evidence_grounding),
            (RuleCategory.RISK_CLASSIFICATION, self._run_risk_classification),
        ]

        for category, dispatcher in dispatchers:
            category_rules = [r for r in all_rules if r.category == category]
            if not category_rules:
                continue
            try:
                issues = dispatcher(summary, category_rules, total_pages=total_pages)
                report.issues.extend(issues)
            except Exception:
                log.exception("Validation category %s failed", category.value)

        # Compute counts
        report.total_issues = len(report.issues)
        report.error_count = sum(1 for i in report.issues if i.severity == IssueSeverity.ERROR)
        report.warning_count = sum(1 for i in report.issues if i.severity == IssueSeverity.WARNING)
        report.info_count = sum(1 for i in report.issues if i.severity == IssueSeverity.INFO)
        report.passed = report.error_count == 0

        return report

    # ── Category dispatchers ────────────────────────────────────────

    @staticmethod
    def _run_data_integrity(
        summary: APSSummary,
        rules: list,
        *,
        total_pages: int = 0,
    ) -> list:
        return check_data_integrity(summary, rules)

    @staticmethod
    def _run_medical_business(
        summary: APSSummary,
        rules: list,
        *,
        total_pages: int = 0,
    ) -> list:
        return check_medical_business(summary, rules)

    @staticmethod
    def _run_evidence_grounding(
        summary: APSSummary,
        rules: list,
        *,
        total_pages: int = 0,
    ) -> list:
        return check_evidence_grounding(summary, rules, total_pages=total_pages)

    @staticmethod
    def _run_risk_classification(
        summary: APSSummary,
        rules: list,
        *,
        total_pages: int = 0,
    ) -> list:
        return check_risk_classification(summary, rules)
