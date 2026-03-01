"""APS validation: rules engine and check modules.

Factory function::

    from scout_ai.domains.aps.validation import create_rules_engine
    engine = create_rules_engine(settings)
    if engine:
        report = engine.validate(aps_summary, total_pages=50)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scout_ai.core.config import AppSettings
    from scout_ai.domains.aps.validation.engine import RulesEngine


def create_rules_engine(settings: AppSettings) -> RulesEngine | None:
    """Create a RulesEngine from application settings, or None if disabled."""
    from scout_ai.domains.aps.validation.engine import RulesEngine

    if not settings.rules.enabled:
        return None

    backend_type = settings.rules.backend

    if backend_type == "file":
        from scout_ai.validation.backends.file_backend import FileRulesBackend

        backend = FileRulesBackend(settings.rules.rules_path)

    elif backend_type == "dynamodb":
        from scout_ai.validation.backends.dynamodb_backend import DynamoDBRulesBackend

        backend = DynamoDBRulesBackend(
            table_name=settings.rules.table_name,
            aws_region=settings.rules.aws_region,
            cache_ttl_seconds=settings.rules.cache_ttl_seconds,
            cache_max_size=settings.rules.cache_max_size,
        )

    elif backend_type == "memory":
        from scout_ai.validation.backends.memory_backend import MemoryRulesBackend

        backend = MemoryRulesBackend()

    else:
        raise ValueError(f"Unknown rules backend: {backend_type!r}")

    return RulesEngine(backend)


__all__ = ["create_rules_engine"]
