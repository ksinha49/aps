"""Circuit breaker hook: prevents cascading failures by short-circuiting after repeated errors."""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strands.hooks.registry import HookRegistry

log = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing — block calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerHook:
    """Strands HookProvider that implements circuit breaker pattern.

    Tracks consecutive failures and transitions through CLOSED → OPEN → HALF_OPEN states.
    When OPEN, raises an error before the model call to prevent wasted API calls.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 60.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._state = CircuitState.CLOSED

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
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
            raise RuntimeError(
                f"Circuit breaker OPEN — {self._failure_count} consecutive failures. "
                f"Retry after {self._recovery_timeout}s."
            )

    def _after_model_call(self, event: Any) -> None:
        error = getattr(event, "error", None)
        if error:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                log.warning(
                    "Circuit breaker → OPEN after %d failures", self._failure_count
                )
        else:
            if self._state == CircuitState.HALF_OPEN:
                log.info("Circuit breaker → CLOSED (successful call in half-open)")
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED
