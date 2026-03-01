"""Risk classification checks: valid tiers, critical-vs-preferred contradictions."""

from __future__ import annotations

from scout_ai.domains.aps.models import APSSummary
from scout_ai.validation.models import Rule, RuleCategory, ValidationIssue


def check_risk_classification(summary: APSSummary, rules: list[Rule]) -> list[ValidationIssue]:
    """Run all risk classification checks."""
    issues: list[ValidationIssue] = []
    rules_by_id = {r.rule_id: r for r in rules if r.category == RuleCategory.RISK_CLASSIFICATION}

    if "RC-001" in rules_by_id:
        issues.extend(_check_valid_tier(summary, rules_by_id["RC-001"]))
    if "RC-002" in rules_by_id:
        issues.extend(_check_critical_vs_tier(summary, rules_by_id["RC-002"]))

    return issues


def _check_valid_tier(summary: APSSummary, rule: Rule) -> list[ValidationIssue]:
    """Validate the risk tier is a recognized value."""
    allowed = set(rule.params.get("allowed_tiers", []))
    tier = summary.risk_classification.tier
    issues: list[ValidationIssue] = []

    if tier and tier not in allowed:
        issues.append(
            ValidationIssue(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                category=rule.category,
                message=f"Unrecognized risk tier '{tier}'",
                field_path="risk_classification.tier",
                actual_value=tier,
                expected_hint=f"One of: {', '.join(sorted(allowed))}",
            )
        )

    return issues


def _check_critical_vs_tier(summary: APSSummary, rule: Rule) -> list[ValidationIssue]:
    """Flag contradiction: CRITICAL finding with Preferred Plus tier."""
    incompatible_tiers = set(rule.params.get("incompatible_tiers", ["Preferred Plus"]))
    trigger_severities = set(rule.params.get("trigger_severities", ["CRITICAL"]))
    tier = summary.risk_classification.tier
    issues: list[ValidationIssue] = []

    if tier not in incompatible_tiers:
        return issues

    # Check all findings and red flags for trigger severities
    has_trigger = False
    trigger_entity = ""
    for section in summary.sections:
        for finding in section.findings:
            if finding.severity in trigger_severities:
                has_trigger = True
                trigger_entity = finding.text[:80]
                break
        if has_trigger:
            break

    if not has_trigger:
        for rf in summary.red_flags:
            if rf.severity in trigger_severities:
                has_trigger = True
                trigger_entity = rf.description[:80]
                break

    if has_trigger:
        issues.append(
            ValidationIssue(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                category=rule.category,
                message=(
                    f"Risk tier '{tier}' contradicts {'/'.join(trigger_severities)} finding: "
                    f"'{trigger_entity}'"
                ),
                field_path="risk_classification.tier",
                entity_name=trigger_entity,
                actual_value=tier,
                expected_hint=(
                    f"Tier should not be {', '.join(incompatible_tiers)} "
                    f"with {'/'.join(trigger_severities)} findings"
                ),
            )
        )

    return issues
