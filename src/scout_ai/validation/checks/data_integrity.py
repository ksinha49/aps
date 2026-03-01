"""Data integrity checks: ICD-10 format, severity enum, lab flags, date formats."""

from __future__ import annotations

import re

from scout_ai.synthesis.models import APSSummary
from scout_ai.validation.models import Rule, RuleCategory, ValidationIssue


def check_data_integrity(summary: APSSummary, rules: list[Rule]) -> list[ValidationIssue]:
    """Run all data integrity checks against the summary."""
    issues: list[ValidationIssue] = []
    rules_by_id = {r.rule_id: r for r in rules if r.category == RuleCategory.DATA_INTEGRITY}

    if "DI-001" in rules_by_id:
        issues.extend(_check_icd10_codes(summary, rules_by_id["DI-001"]))
    if "DI-002" in rules_by_id:
        issues.extend(_check_severity_values(summary, rules_by_id["DI-002"]))
    if "DI-003" in rules_by_id:
        issues.extend(_check_lab_flags(summary, rules_by_id["DI-003"]))
    if "DI-004" in rules_by_id:
        issues.extend(_check_date_formats(summary, rules_by_id["DI-004"]))

    return issues


def _check_icd10_codes(summary: APSSummary, rule: Rule) -> list[ValidationIssue]:
    """Validate ICD-10 codes on conditions match the expected regex pattern."""
    pattern = re.compile(rule.params.get("pattern", r"^[A-Z]\d{2}(\.\d{1,4})?$"))
    issues: list[ValidationIssue] = []

    for section in summary.sections:
        for condition in section.conditions:
            code = condition.icd10_code
            if not code:
                continue
            if not pattern.match(code):
                issues.append(
                    ValidationIssue(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        category=rule.category,
                        message=f"Invalid ICD-10 code '{code}' on condition '{condition.name}'",
                        section_key=section.section_key,
                        field_path="conditions[].icd10_code",
                        entity_name=condition.name,
                        actual_value=code,
                        expected_hint="Format: A00-Z99 with optional .0-.9999",
                    )
                )

    return issues


def _check_severity_values(summary: APSSummary, rule: Rule) -> list[ValidationIssue]:
    """Validate severity values on findings and red flags."""
    allowed = set(rule.params.get("allowed", []))
    issues: list[ValidationIssue] = []

    for section in summary.sections:
        for finding in section.findings:
            if finding.severity not in allowed:
                issues.append(
                    ValidationIssue(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        category=rule.category,
                        message=f"Invalid severity '{finding.severity}' on finding",
                        section_key=section.section_key,
                        field_path="findings[].severity",
                        entity_name=finding.text[:80],
                        actual_value=finding.severity,
                        expected_hint=f"One of: {', '.join(sorted(allowed))}",
                    )
                )

    for red_flag in summary.red_flags:
        if red_flag.severity not in allowed:
            issues.append(
                ValidationIssue(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    category=rule.category,
                    message=f"Invalid severity '{red_flag.severity}' on red flag",
                    field_path="red_flags[].severity",
                    entity_name=red_flag.description[:80],
                    actual_value=red_flag.severity,
                    expected_hint=f"One of: {', '.join(sorted(allowed))}",
                )
            )

    return issues


def _check_lab_flags(summary: APSSummary, rule: Rule) -> list[ValidationIssue]:
    """Validate lab result flag values."""
    allowed = set(rule.params.get("allowed", []))
    issues: list[ValidationIssue] = []

    for section in summary.sections:
        for lab in section.lab_results:
            if lab.flag not in allowed:
                issues.append(
                    ValidationIssue(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        category=rule.category,
                        message=f"Invalid lab flag '{lab.flag}' on '{lab.test_name}'",
                        section_key=section.section_key,
                        field_path="lab_results[].flag",
                        entity_name=lab.test_name,
                        actual_value=lab.flag,
                        expected_hint=f"One of: {', '.join(repr(v) for v in sorted(allowed))}",
                    )
                )

    return issues


def _check_date_formats(summary: APSSummary, rule: Rule) -> list[ValidationIssue]:
    """Validate date fields match accepted formats."""
    raw_patterns = rule.params.get("patterns", [])
    patterns = [re.compile(p) for p in raw_patterns]
    issues: list[ValidationIssue] = []

    def _validate_date(value: str, entity: str, field_path: str, section_key: str) -> None:
        if not value:
            return
        if any(p.match(value) for p in patterns):
            return
        issues.append(
            ValidationIssue(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                category=rule.category,
                message=f"Unrecognized date format '{value}' on '{entity}'",
                section_key=section_key,
                field_path=field_path,
                entity_name=entity,
                actual_value=value,
                expected_hint="MM/DD/YYYY, MM/YYYY, YYYY-MM-DD, or YYYY",
            )
        )

    for section in summary.sections:
        for cond in section.conditions:
            _validate_date(cond.onset_date, cond.name, "conditions[].onset_date", section.section_key)
        for lab in section.lab_results:
            _validate_date(lab.date, lab.test_name, "lab_results[].date", section.section_key)
        for med in section.medications:
            _validate_date(med.start_date, med.name, "medications[].start_date", section.section_key)
        for enc in section.encounters:
            _validate_date(enc.date, enc.provider or "encounter", "encounters[].date", section.section_key)

    return issues
