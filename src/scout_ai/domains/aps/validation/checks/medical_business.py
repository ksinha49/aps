"""Medical business rule checks: HbA1c, BMI, auto-critical, controlled substances."""

from __future__ import annotations

import re

from scout_ai.domains.aps.models import APSSummary
from scout_ai.validation.models import Rule, RuleCategory, ValidationIssue

# Severity ordering for comparison
_SEVERITY_ORDER = {
    "INFORMATIONAL": 0,
    "MINOR": 1,
    "MODERATE": 2,
    "SIGNIFICANT": 3,
    "CRITICAL": 4,
}


def check_medical_business(summary: APSSummary, rules: list[Rule]) -> list[ValidationIssue]:
    """Run all medical business checks against the summary."""
    issues: list[ValidationIssue] = []
    rules_by_id = {r.rule_id: r for r in rules if r.category == RuleCategory.MEDICAL_BUSINESS}

    if "MB-001" in rules_by_id:
        issues.extend(_check_hba1c_severity(summary, rules_by_id["MB-001"]))
    if "MB-002" in rules_by_id:
        issues.extend(_check_bmi_risk_factor(summary, rules_by_id["MB-002"]))
    if "MB-003" in rules_by_id:
        issues.extend(_check_auto_critical_conditions(summary, rules_by_id["MB-003"]))
    if "MB-004" in rules_by_id:
        issues.extend(_check_controlled_substances(summary, rules_by_id["MB-004"]))

    return issues


def _parse_numeric(value: str) -> float | None:
    """Extract the first numeric value from a string like '7.2 %' or '32.1'."""
    match = re.search(r"(\d+\.?\d*)", value)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _check_hba1c_severity(summary: APSSummary, rule: Rule) -> list[ValidationIssue]:
    """Flag HbA1c values above threshold with insufficient severity."""
    test_names = {n.lower() for n in rule.params.get("test_names", [])}
    threshold = rule.params.get("threshold", 7.0)
    min_severity = rule.params.get("min_severity", "MODERATE")
    min_severity_rank = _SEVERITY_ORDER.get(min_severity, 2)
    issues: list[ValidationIssue] = []

    for section in summary.sections:
        for lab in section.lab_results:
            if lab.test_name.lower() not in test_names:
                continue
            numeric = _parse_numeric(lab.value)
            if numeric is None or numeric <= threshold:
                continue

            # Check if there's a related finding with adequate severity
            section_findings_max = max(
                (_SEVERITY_ORDER.get(f.severity, 0) for f in section.findings),
                default=0,
            )
            if section_findings_max < min_severity_rank:
                issues.append(
                    ValidationIssue(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        category=rule.category,
                        message=(
                            f"{lab.test_name} value {lab.value} exceeds threshold {threshold}; "
                            f"expected severity >= {min_severity}"
                        ),
                        section_key=section.section_key,
                        field_path="lab_results[].value",
                        entity_name=lab.test_name,
                        actual_value=lab.value,
                        expected_hint=f"Severity >= {min_severity}",
                    )
                )

    return issues


def _check_bmi_risk_factor(summary: APSSummary, rule: Rule) -> list[ValidationIssue]:
    """Flag BMI above threshold not listed in risk factors."""
    vital_names = {n.lower() for n in rule.params.get("vital_names", [])}
    threshold = rule.params.get("threshold", 30.0)
    issues: list[ValidationIssue] = []

    risk_factors_lower = [rf.lower() for rf in summary.risk_factors]

    for section in summary.sections:
        for vital in section.vital_signs:
            if vital.name.lower() not in vital_names:
                continue
            numeric = _parse_numeric(vital.value)
            if numeric is None or numeric <= threshold:
                continue

            # Check if BMI or obesity is mentioned in risk factors
            bmi_in_risks = any(
                "bmi" in rf or "obesity" in rf or "obese" in rf
                for rf in risk_factors_lower
            )
            if not bmi_in_risks:
                issues.append(
                    ValidationIssue(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        category=rule.category,
                        message=(
                            f"BMI {vital.value} exceeds {threshold} but is not reflected in risk_factors"
                        ),
                        section_key=section.section_key,
                        field_path="vital_signs[].value",
                        entity_name=vital.name,
                        actual_value=vital.value,
                        expected_hint="Add BMI/obesity to risk_factors",
                    )
                )

    return issues


def _check_auto_critical_conditions(summary: APSSummary, rule: Rule) -> list[ValidationIssue]:
    """Flag conditions matching serious disease patterns that aren't CRITICAL."""
    raw_patterns = rule.params.get("condition_patterns", [])
    patterns = [re.compile(p) for p in raw_patterns]
    issues: list[ValidationIssue] = []

    for section in summary.sections:
        for condition in section.conditions:
            matched = any(p.search(condition.name) for p in patterns)
            if not matched:
                continue
            if condition.severity != "CRITICAL":
                issues.append(
                    ValidationIssue(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        category=rule.category,
                        message=(
                            f"Condition '{condition.name}' matches auto-critical pattern "
                            f"but has severity '{condition.severity}'"
                        ),
                        section_key=section.section_key,
                        field_path="conditions[].severity",
                        entity_name=condition.name,
                        actual_value=condition.severity,
                        expected_hint="CRITICAL",
                    )
                )

    return issues


def _check_controlled_substances(summary: APSSummary, rule: Rule) -> list[ValidationIssue]:
    """Flag concurrent controlled substances without a red flag."""
    raw_patterns = rule.params.get("controlled_patterns", [])
    patterns = [re.compile(p) for p in raw_patterns]
    min_concurrent = rule.params.get("min_concurrent", 2)
    issues: list[ValidationIssue] = []

    # Collect all medications across sections
    controlled_names: list[str] = []
    for section in summary.sections:
        for med in section.medications:
            if any(p.search(med.name) for p in patterns):
                controlled_names.append(med.name)

    if len(controlled_names) >= min_concurrent:
        # Check if a red flag exists for this
        red_flag_exists = any(
            "controlled" in rf.description.lower() or "substance" in rf.description.lower()
            for rf in summary.red_flags
        )
        if not red_flag_exists:
            issues.append(
                ValidationIssue(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    category=rule.category,
                    message=(
                        f"{len(controlled_names)} concurrent controlled substances detected "
                        f"({', '.join(controlled_names[:5])}) but no corresponding red flag found"
                    ),
                    field_path="medications[]",
                    entity_name=", ".join(controlled_names[:3]),
                    actual_value=str(len(controlled_names)),
                    expected_hint=f"Add red flag when >= {min_concurrent} controlled substances",
                )
            )

    return issues
