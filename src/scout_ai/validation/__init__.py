"""Validation module: rules engine, backends, and check modules.

Generic models (``Rule``, ``ValidationIssue``, ``ValidationReport``) and
backends (``file``, ``dynamodb``, ``memory``) remain here.
APS-specific logic (engine, checks) lives in ``domains.aps.validation``.

Factory function::

    from scout_ai.validation import create_rules_engine
    engine = create_rules_engine(settings)
    if engine:
        report = engine.validate(aps_summary, total_pages=50)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scout_ai.validation.models import (
    IssueSeverity,
    Rule,
    RuleCategory,
    RuleTarget,
    ValidationIssue,
    ValidationReport,
)

if TYPE_CHECKING:
    from scout_ai.core.config import AppSettings
    from scout_ai.domains.aps.validation.engine import RulesEngine


def create_rules_engine(settings: AppSettings) -> RulesEngine | None:
    """Create a RulesEngine from application settings, or None if disabled.

    Re-export shim — delegates to ``domains.aps.validation.create_rules_engine``.
    """
    from scout_ai.domains.aps.validation import create_rules_engine as _create

    return _create(settings)


__all__ = [
    "Rule",
    "RuleCategory",
    "RuleTarget",
    "IssueSeverity",
    "ValidationIssue",
    "ValidationReport",
    "create_rules_engine",
]
