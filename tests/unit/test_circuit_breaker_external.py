"""Tests for pluggable circuit breaker state store and integration with CircuitBreakerHook."""

from __future__ import annotations

from unittest.mock import MagicMock

from scout_ai.hooks.circuit_breaker_hook import CircuitBreakerHook, CircuitState
from scout_ai.hooks.circuit_breaker_store import IBreakerStore, MemoryBreakerStore

# ---------------------------------------------------------------------------
# MemoryBreakerStore unit tests
# ---------------------------------------------------------------------------


class TestMemoryBreakerStore:
    def test_memory_store_records_failures(self) -> None:
        store = MemoryBreakerStore()
        count1 = store.record_failure("k1")
        count2 = store.record_failure("k1")

        assert count1 == 1
        assert count2 == 2
        assert store.get_failure_count("k1") == 2

    def test_memory_store_reset(self) -> None:
        store = MemoryBreakerStore()
        store.record_failure("k1")
        store.record_failure("k1")
        store.reset("k1")

        assert store.get_failure_count("k1") == 0

    def test_memory_store_last_failure_time(self) -> None:
        store = MemoryBreakerStore()
        # Before any failure, time is 0.0
        assert store.get_last_failure_time("k1") == 0.0

        store.record_failure("k1")
        t = store.get_last_failure_time("k1")
        assert t > 0.0

    def test_memory_store_independent_keys(self) -> None:
        store = MemoryBreakerStore()
        store.record_failure("a")
        store.record_failure("a")
        store.record_failure("b")

        assert store.get_failure_count("a") == 2
        assert store.get_failure_count("b") == 1

        store.reset("a")
        assert store.get_failure_count("a") == 0
        assert store.get_failure_count("b") == 1

    def test_memory_store_satisfies_protocol(self) -> None:
        store = MemoryBreakerStore()
        assert isinstance(store, IBreakerStore)


# ---------------------------------------------------------------------------
# CircuitBreakerHook + store integration tests
# ---------------------------------------------------------------------------


def _make_error_event() -> MagicMock:
    event = MagicMock()
    event.error = RuntimeError("boom")
    return event


def _make_success_event() -> MagicMock:
    event = MagicMock(spec=[])  # no 'error' attribute
    return event


class TestCircuitBreakerWithStore:
    def test_circuit_breaker_uses_store(self) -> None:
        store = MemoryBreakerStore()
        hook = CircuitBreakerHook(store=store, breaker_key="svc-a")

        assert hook.state == CircuitState.CLOSED
        assert store.get_failure_count("svc-a") == 0

    def test_breaker_opens_after_threshold_via_store(self) -> None:
        store = MemoryBreakerStore()
        hook = CircuitBreakerHook(
            failure_threshold=3,
            store=store,
            breaker_key="svc-b",
        )

        error_event = _make_error_event()
        for _ in range(3):
            hook._after_model_call(error_event)

        assert hook.state == CircuitState.OPEN
        assert store.get_failure_count("svc-b") == 3

    def test_breaker_reset_clears_store(self) -> None:
        store = MemoryBreakerStore()
        hook = CircuitBreakerHook(
            failure_threshold=3,
            store=store,
            breaker_key="svc-c",
        )

        error_event = _make_error_event()
        for _ in range(3):
            hook._after_model_call(error_event)

        assert hook.state == CircuitState.OPEN
        assert store.get_failure_count("svc-c") == 3

        hook.reset()
        assert hook.state == CircuitState.CLOSED
        assert store.get_failure_count("svc-c") == 0

    def test_default_store_is_memory(self) -> None:
        hook = CircuitBreakerHook()
        assert isinstance(hook._store, MemoryBreakerStore)

    def test_success_resets_store_count(self) -> None:
        store = MemoryBreakerStore()
        hook = CircuitBreakerHook(
            failure_threshold=5,
            store=store,
            breaker_key="svc-d",
        )

        error_event = _make_error_event()
        hook._after_model_call(error_event)
        hook._after_model_call(error_event)
        assert store.get_failure_count("svc-d") == 2

        success_event = _make_success_event()
        hook._after_model_call(success_event)
        assert store.get_failure_count("svc-d") == 0

    def test_shared_store_across_hooks(self) -> None:
        """Two hooks sharing a store with different keys stay independent."""
        store = MemoryBreakerStore()
        hook_a = CircuitBreakerHook(failure_threshold=2, store=store, breaker_key="a")
        hook_b = CircuitBreakerHook(failure_threshold=2, store=store, breaker_key="b")

        error_event = _make_error_event()
        hook_a._after_model_call(error_event)
        hook_a._after_model_call(error_event)

        assert hook_a.state == CircuitState.OPEN
        assert hook_b.state == CircuitState.CLOSED
        assert store.get_failure_count("a") == 2
        assert store.get_failure_count("b") == 0
