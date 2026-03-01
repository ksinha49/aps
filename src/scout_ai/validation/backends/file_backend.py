"""File-backed rules backend — loads rules from YAML or JSON on disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from scout_ai.validation.models import (
    IssueSeverity,
    Rule,
    RuleCategory,
    RuleTarget,
)

log = logging.getLogger(__name__)


class FileRulesBackend:
    """Loads rules from a YAML or JSON file on disk.

    The file is lazy-loaded on first ``list_rules()`` call.
    PyYAML is only required for ``.yaml`` / ``.yml`` files.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._rules: dict[str, Rule] | None = None
        self._version: int = 1

    def list_rules(
        self,
        *,
        category: RuleCategory | None = None,
        enabled_only: bool = True,
        version: int | None = None,
        lob: str = "*",
    ) -> list[Rule]:
        """Return rules from the file, optionally filtered."""
        self._ensure_loaded()
        assert self._rules is not None
        result: list[Rule] = []
        for rule in self._rules.values():
            if enabled_only and not rule.enabled:
                continue
            if category is not None and rule.category != category:
                continue
            result.append(rule)
        return result

    def get_rule(self, rule_id: str, *, version: int | None = None) -> Rule:
        """Get a single rule by ID."""
        self._ensure_loaded()
        assert self._rules is not None
        if rule_id not in self._rules:
            raise KeyError(f"Rule {rule_id!r} not found in {self._path}")
        return self._rules[rule_id]

    def get_version(self) -> int:
        """Return the ruleset version."""
        self._ensure_loaded()
        return self._version

    def _ensure_loaded(self) -> None:
        """Lazy-load the rules file on first access."""
        if self._rules is not None:
            return

        if not self._path.exists():
            raise FileNotFoundError(f"Rules file not found: {self._path}")

        raw_text = self._path.read_text(encoding="utf-8")

        if self._path.suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required for YAML rules files. "
                    "Install with: pip install scout-ai[rules]"
                ) from exc
            data = yaml.safe_load(raw_text)
        else:
            data = json.loads(raw_text)

        self._parse(data)

    def _parse(self, data: dict[str, Any]) -> None:
        """Parse the raw dict into Rule objects."""
        self._version = data.get("version", 1)
        self._rules = {}

        for rule_data in data.get("rules", []):
            rule = Rule(
                rule_id=rule_data["rule_id"],
                name=rule_data["name"],
                description=rule_data.get("description", ""),
                category=RuleCategory(rule_data["category"]),
                target=RuleTarget(rule_data["target"]),
                severity=IssueSeverity(rule_data.get("severity", "warning")),
                enabled=rule_data.get("enabled", True),
                params=rule_data.get("params", {}),
                version=rule_data.get("version", self._version),
            )
            self._rules[rule.rule_id] = rule

        log.info("Loaded %d rules from %s (version %d)", len(self._rules), self._path, self._version)
