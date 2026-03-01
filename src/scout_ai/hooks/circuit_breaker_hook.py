"""Circuit breaker hook: prevents cascading failures by short-circuiting after repeated errors."""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import TYPE_CHECKING, Any

from scout_ai.hooks.circuit_breaker_store import IBreakerStore, MemoryBreakerStore

if TYPE_CHECKING:
    from strands.hooks.registry import HookRegistry

log = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing — block calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerHook:
    """Strands HookProvider that implements circuit breaker pattern.

    Tracks consecutive failures and transitions through CLOSED -> OPEN -> HALF_OPEN states.
    When OPEN, raises an error before the model call to prevent wasted API calls.

    Failure counts and timestamps are delegated to an ``IBreakerStore`` so that
    state can optionally be shared across processes (e.g. via Redis or DynamoDB).
    When no store is provided, an in-process ``MemoryBreakerStore`` is used.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 60.0,
        store: IBreakerStore | None = None,
        breaker_key: str = "default",
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._store: IBreakerStore = store if store is not None else MemoryBreakerStore()
        self._breaker_key = breaker_key
        self._state = CircuitState.CLOSED

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            last_failure = self._store.get_last_failure_time(self._breaker_key)
            elapsed = time.monotonic() - last_failure
            if elapsed >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                log.info("Circuit breaker → HALF_OPEN (recovery timeout elapsed)")
        return self._state

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        from strands.hooks.events import AfterModelCallEvent, BeforeModelCallEvent

        registry.add_callback(BeforeModelCallEvent, self._before_model_call)
        registry.add_callback(AfterModelCallEvent, self._after_model_call)

    def _before_model_call(self, event: Any) -> None:
        current = self.state
        if current == CircuitState.OPEN:
            failure_count = self._store.get_failure_count(self._breaker_key)
            raise RuntimeError(
                f"Circuit breaker OPEN — {failure_count} consecutive failures. "
                f"Retry after {self._recovery_timeout}s."
            )

    def _after_model_call(self, event: Any) -> None:
        error = getattr(event, "error", None)
        if error:
            count = self._store.record_failure(self._breaker_key)
            if count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                log.warning(
                    "Circuit breaker → OPEN after %d failures", count
                )
        else:
            if self._state == CircuitState.HALF_OPEN:
                log.info("Circuit breaker → CLOSED (successful call in half-open)")
            self._store.reset(self._breaker_key)
            self._state = CircuitState.CLOSED

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED."""
        self._store.reset(self._breaker_key)
        self._state = CircuitState.CLOSED
