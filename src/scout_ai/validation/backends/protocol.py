"""Rules backend protocol — defines the contract all backends implement."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from scout_ai.validation.models import Rule, RuleCategory


@runtime_checkable
class IRulesBackend(Protocol):
    """Protocol for rules storage backends (file, DynamoDB, memory)."""

    def list_rules(
        self,
        *,
        category: RuleCategory | None = None,
        enabled_only: bool = True,
        version: int | None = None,
        lob: str = "*",
    ) -> list[Rule]:
        """Return rules, optionally filtered by category and enabled status."""
        ...

    def get_rule(self, rule_id: str, *, version: int | None = None) -> Rule:
        """Get a single rule by ID. Raises KeyError if not found."""
        ...

    def get_version(self) -> int:
        """Return the current ruleset version number."""
        ...
