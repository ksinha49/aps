"""Evidence grounding checks: citation requirements, page ranges, uncited severity caps."""

from __future__ import annotations

from scout_ai.synthesis.models import APSSummary
from scout_ai.validation.models import Rule, RuleCategory, ValidationIssue

# Severity ordering for cap comparison
_SEVERITY_ORDER = {
    "INFORMATIONAL": 0,
    "MINOR": 1,
    "MODERATE": 2,
    "SIGNIFICANT": 3,
    "CRITICAL": 4,
}


def check_evidence_grounding(
    summary: APSSummary,
    rules: list[Rule],
    *,
    total_pages: int = 0,
) -> list[ValidationIssue]:
    """Run all evidence grounding checks."""
    issues: list[ValidationIssue] = []
    rules_by_id = {r.rule_id: r for r in rules if r.category == RuleCategory.EVIDENCE_GROUNDING}

    if "EG-001" in rules_by_id:
        issues.extend(_check_citations_required(summary, rules_by_id["EG-001"]))
    if "EG-002" in rules_by_id and total_pages > 0:
        issues.extend(_check_citation_page_range(summary, rules_by_id["EG-002"], total_pages))
    if "EG-003" in rules_by_id:
        issues.extend(_check_uncited_severity_cap(summary, rules_by_id["EG-003"]))

    return issues


def _check_citations_required(summary: APSSummary, rule: Rule) -> list[ValidationIssue]:
    """CRITICAL/SIGNIFICANT findings must have at least one citation."""
    require_for = set(rule.params.get("require_citation_for", ["CRITICAL", "SIGNIFICANT"]))
    issues: list[ValidationIssue] = []

    for section in summary.sections:
        for finding in section.findings:
            if finding.severity not in require_for:
                continue
            if not finding.citations:
                issues.append(
                    ValidationIssue(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        category=rule.category,
                        message=(
                            f"{finding.severity} finding has no citations: "
                            f"'{finding.text[:80]}'"
                        ),
                        section_key=section.section_key,
                        field_path="findings[].citations",
                        entity_name=finding.text[:80],
                        actual_value="0 citations",
                        expected_hint="At least 1 citation required",
                    )
                )

    return issues


def _check_citation_page_range(
    summary: APSSummary,
    rule: Rule,
    total_pages: int,
) -> list[ValidationIssue]:
    """Citation page numbers must be within [1, total_pages]."""
    issues: list[ValidationIssue] = []

    for section in summary.sections:
        for finding in section.findings:
            for citation in finding.citations:
                if citation.page_number < 1 or citation.page_number > total_pages:
                    issues.append(
                        ValidationIssue(
                            rule_id=rule.rule_id,
                            rule_name=rule.name,
                            severity=rule.severity,
                            category=rule.category,
                            message=(
                                f"Citation page {citation.page_number} is out of range "
                                f"[1, {total_pages}] for finding '{finding.text[:60]}'"
                            ),
                            section_key=section.section_key,
                            field_path="findings[].citations[].page_number",
                            entity_name=finding.text[:80],
                            actual_value=str(citation.page_number),
                            expected_hint=f"1 to {total_pages}",
                        )
                    )
        # Also check conditions, medications, etc. citations
        for condition in section.conditions:
            for citation in condition.citations:
                if citation.page_number < 1 or citation.page_number > total_pages:
                    issues.append(
                        ValidationIssue(
                            rule_id=rule.rule_id,
                            rule_name=rule.name,
                            severity=rule.severity,
                            category=rule.category,
                            message=(
                                f"Citation page {citation.page_number} out of range "
                                f"[1, {total_pages}] for condition '{condition.name}'"
                            ),
                            section_key=section.section_key,
                            field_path="conditions[].citations[].page_number",
                            entity_name=condition.name,
                            actual_value=str(citation.page_number),
                            expected_hint=f"1 to {total_pages}",
                        )
                    )

    return issues


def _check_uncited_severity_cap(summary: APSSummary, rule: Rule) -> list[ValidationIssue]:
    """Flag findings without citations that exceed the severity cap."""
    max_severity_name = rule.params.get("max_uncited_severity", "MINOR")
    max_rank = _SEVERITY_ORDER.get(max_severity_name, 1)
    issues: list[ValidationIssue] = []

    for section in summary.sections:
        for finding in section.findings:
            if finding.citations:
                continue
            finding_rank = _SEVERITY_ORDER.get(finding.severity, 0)
            if finding_rank > max_rank:
                issues.append(
                    ValidationIssue(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        category=rule.category,
                        message=(
                            f"Uncited finding has severity '{finding.severity}' which exceeds "
                            f"cap '{max_severity_name}': '{finding.text[:60]}'"
                        ),
                        section_key=section.section_key,
                        field_path="findings[].severity",
                        entity_name=finding.text[:80],
                        actual_value=finding.severity,
                        expected_hint=f"<= {max_severity_name} when uncited",
                    )
                )

    return issues
