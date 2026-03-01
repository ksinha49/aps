"""In-memory rules backend for testing."""

from __future__ import annotations

from scout_ai.validation.models import Rule, RuleCategory


class MemoryRulesBackend:
    """Dict-backed rules backend for unit tests."""

    def __init__(self, rules: list[Rule] | None = None, *, version: int = 1) -> None:
        self._rules = {r.rule_id: r for r in (rules or [])}
        self._version = version

    def list_rules(
        self,
        *,
        category: RuleCategory | None = None,
        enabled_only: bool = True,
        version: int | None = None,
        lob: str = "*",
    ) -> list[Rule]:
        """Return rules, optionally filtered."""
        result: list[Rule] = []
        for rule in self._rules.values():
            if enabled_only and not rule.enabled:
                continue
            if category is not None and rule.category != category:
                continue
            result.append(rule)
        return result

    def get_rule(self, rule_id: str, *, version: int | None = None) -> Rule:
        """Get a rule by ID."""
        if rule_id not in self._rules:
            raise KeyError(f"Rule {rule_id!r} not found")
        return self._rules[rule_id]

    def get_version(self) -> int:
        """Return the ruleset version."""
        return self._version
