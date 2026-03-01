"""Tests for MemoryRulesBackend."""

from __future__ import annotations

import pytest

from scout_ai.validation.backends.memory_backend import MemoryRulesBackend
from scout_ai.validation.models import IssueSeverity, Rule, RuleCategory, RuleTarget


def _rule(rule_id: str, category: RuleCategory, *, enabled: bool = True) -> Rule:
    return Rule(
        rule_id=rule_id,
        name=f"Rule {rule_id}",
        description="test",
        category=category,
        target=RuleTarget.FINDING,
        severity=IssueSeverity.WARNING,
        enabled=enabled,
    )


class TestMemoryRulesBackend:
    def test_list_empty(self) -> None:
        backend = MemoryRulesBackend()
        assert backend.list_rules() == []

    def test_list_all(self) -> None:
        rules = [
            _rule("A", RuleCategory.DATA_INTEGRITY),
            _rule("B", RuleCategory.EVIDENCE_GROUNDING),
        ]
        backend = MemoryRulesBackend(rules)
        assert len(backend.list_rules()) == 2

    def test_filter_by_category(self) -> None:
        rules = [
            _rule("A", RuleCategory.DATA_INTEGRITY),
            _rule("B", RuleCategory.EVIDENCE_GROUNDING),
        ]
        backend = MemoryRulesBackend(rules)
        result = backend.list_rules(category=RuleCategory.DATA_INTEGRITY)
        assert len(result) == 1
        assert result[0].rule_id == "A"

    def test_filter_enabled_only(self) -> None:
        rules = [
            _rule("A", RuleCategory.DATA_INTEGRITY, enabled=True),
            _rule("B", RuleCategory.DATA_INTEGRITY, enabled=False),
        ]
        backend = MemoryRulesBackend(rules)
        assert len(backend.list_rules()) == 1
        assert len(backend.list_rules(enabled_only=False)) == 2

    def test_get_rule(self) -> None:
        rules = [_rule("A", RuleCategory.DATA_INTEGRITY)]
        backend = MemoryRulesBackend(rules)
        assert backend.get_rule("A").rule_id == "A"

    def test_get_rule_not_found(self) -> None:
        backend = MemoryRulesBackend()
        with pytest.raises(KeyError):
            backend.get_rule("MISSING")

    def test_version(self) -> None:
        backend = MemoryRulesBackend(version=5)
        assert backend.get_version() == 5
