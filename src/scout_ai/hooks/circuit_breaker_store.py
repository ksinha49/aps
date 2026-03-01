"""Pluggable circuit breaker state store protocol and in-memory implementation."""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class IBreakerStore(Protocol):
    """Protocol for circuit breaker state storage backends.

    Implementations track consecutive failure counts and timestamps per breaker key,
    allowing state to be externalized (e.g. Redis, DynamoDB) for cross-process
    circuit breaker coordination.
    """

    def record_failure(self, key: str) -> int:
        """Record a failure and return the new consecutive failure count."""
        ...

    def get_failure_count(self, key: str) -> int:
        """Get the current consecutive failure count for a key."""
        ...

    def get_last_failure_time(self, key: str) -> float:
        """Get the monotonic timestamp of the most recent failure."""
        ...

    def reset(self, key: str) -> None:
        """Reset the failure count to zero for a key."""
        ...


class MemoryBreakerStore:
    """In-process circuit breaker state store backed by plain dicts.

    Uses ``time.monotonic()`` for timestamps so values are immune to wall-clock
    adjustments.
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._times: dict[str, float] = {}

    def record_failure(self, key: str) -> int:
        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        self._times[key] = time.monotonic()
        return count

    def get_failure_count(self, key: str) -> int:
        return self._counts.get(key, 0)

    def get_last_failure_time(self, key: str) -> float:
        return self._times.get(key, 0.0)

    def reset(self, key: str) -> None:
        self._counts[key] = 0
        self._times[key] = 0.0
