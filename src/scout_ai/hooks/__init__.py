"""Strands lifecycle hooks: audit, cost, checkpoint, circuit breaker, dead letter, tracing, logging, run tracker."""

from __future__ import annotations

from scout_ai.hooks.audit_hook import AuditHook
from scout_ai.hooks.checkpoint_hook import CheckpointHook
from scout_ai.hooks.circuit_breaker_hook import CircuitBreakerHook, CircuitState
from scout_ai.hooks.circuit_breaker_store import IBreakerStore, MemoryBreakerStore
from scout_ai.hooks.cost_hook import CostHook, UsageSummary, get_current_usage, reset_usage
from scout_ai.hooks.dead_letter_hook import DeadLetterHook
from scout_ai.hooks.logging_config import setup_logging
from scout_ai.hooks.run_tracker import end_run, get_current_run, start_run, track_stage
from scout_ai.hooks.tracing import setup_tracing

__all__ = [
    "AuditHook",
    "CheckpointHook",
    "CircuitBreakerHook",
    "CircuitState",
    "IBreakerStore",
    "MemoryBreakerStore",
    "CostHook",
    "DeadLetterHook",
    "UsageSummary",
    "create_resilience_hooks",
    "end_run",
    "get_current_run",
    "get_current_usage",
    "reset_usage",
    "setup_logging",
    "setup_tracing",
    "start_run",
    "track_stage",
]


def create_resilience_hooks(
    resilience_config: object,
    persistence_backend: object | None = None,
    breaker_store: IBreakerStore | None = None,
) -> list[object]:
    """Create resilience hooks wired from ``ResilienceConfig``.

    Args:
        resilience_config: A ``ResilienceConfig`` instance (or any object with
            the expected attributes).
        persistence_backend: Optional ``IPersistenceBackend`` for checkpoint
            and dead-letter hooks.  When ``None`` those hooks are omitted.
        breaker_store: Optional ``IBreakerStore`` for the circuit breaker.

    Returns:
        List of hook instances ready to pass to ``Agent(hooks=...)``.
    """
    hooks: list[object] = []

    hooks.append(
        CircuitBreakerHook(
            failure_threshold=resilience_config.circuit_breaker_failure_threshold,  # type: ignore[attr-defined]
            recovery_timeout_seconds=resilience_config.circuit_breaker_recovery_timeout,  # type: ignore[attr-defined]
            store=breaker_store,
        )
    )

    if persistence_backend is not None:
        hooks.append(
            CheckpointHook(
                backend=persistence_backend,  # type: ignore[arg-type]
                key_prefix=resilience_config.checkpoint_key_prefix,  # type: ignore[attr-defined]
            )
        )
        hooks.append(
            DeadLetterHook(
                backend=persistence_backend,  # type: ignore[arg-type]
                key_prefix=resilience_config.dead_letter_key_prefix,  # type: ignore[attr-defined]
            )
        )

    return hooks
