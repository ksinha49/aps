"""Tests for CostHook cache metrics tracking."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pageindex_rag.hooks.cost_hook import CostHook, UsageSummary, get_current_usage, reset_usage


class TestUsageSummary:
    def test_defaults(self) -> None:
        s = UsageSummary()
        assert s.prompt_tokens == 0
        assert s.completion_tokens == 0
        assert s.cached_tokens == 0
        assert s.cache_creation_tokens == 0
        assert s.call_count == 0
        assert s.cache_hit_count == 0
        assert s.cache_miss_count == 0

    def test_total_tokens(self) -> None:
        s = UsageSummary(prompt_tokens=100, completion_tokens=50)
        assert s.total_tokens == 150

    def test_cache_hit_rate_no_calls(self) -> None:
        s = UsageSummary()
        assert s.cache_hit_rate == 0.0

    def test_cache_hit_rate(self) -> None:
        s = UsageSummary(cache_hit_count=9, cache_miss_count=1)
        assert s.cache_hit_rate == pytest.approx(0.9)

    def test_cache_hit_rate_all_hits(self) -> None:
        s = UsageSummary(cache_hit_count=10, cache_miss_count=0)
        assert s.cache_hit_rate == pytest.approx(1.0)

    def test_estimated_savings_ratio(self) -> None:
        s = UsageSummary(prompt_tokens=1000, cached_tokens=900)
        # 900 * 0.9 / 1000 = 0.81
        assert s.estimated_savings_ratio == pytest.approx(0.81)

    def test_estimated_savings_no_prompt_tokens(self) -> None:
        s = UsageSummary(prompt_tokens=0, cached_tokens=0)
        assert s.estimated_savings_ratio == 0.0

    def test_estimated_savings_no_cache(self) -> None:
        s = UsageSummary(prompt_tokens=1000, cached_tokens=0)
        assert s.estimated_savings_ratio == 0.0


class TestCostHookCacheTracking:
    def setup_method(self) -> None:
        reset_usage()

    def test_cache_hit_tracked(self) -> None:
        hook = CostHook()
        event = MagicMock()
        event.usage = {
            "inputTokens": 1000,
            "outputTokens": 200,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 0,
        }

        hook._on_model_call(event)
        summary = get_current_usage()

        assert summary.prompt_tokens == 1000
        assert summary.completion_tokens == 200
        assert summary.cached_tokens == 800
        assert summary.cache_creation_tokens == 0
        assert summary.cache_hit_count == 1
        assert summary.cache_miss_count == 0
        assert summary.call_count == 1

    def test_cache_miss_tracked(self) -> None:
        hook = CostHook()
        event = MagicMock()
        event.usage = {
            "inputTokens": 1000,
            "outputTokens": 200,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 1000,
        }

        hook._on_model_call(event)
        summary = get_current_usage()

        assert summary.cache_creation_tokens == 1000
        assert summary.cache_hit_count == 0
        assert summary.cache_miss_count == 1

    def test_multiple_calls_accumulate(self) -> None:
        hook = CostHook()

        # First call: cache write
        event1 = MagicMock()
        event1.usage = {
            "inputTokens": 1000,
            "outputTokens": 100,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 900,
        }
        hook._on_model_call(event1)

        # Second call: cache hit
        event2 = MagicMock()
        event2.usage = {
            "inputTokens": 1000,
            "outputTokens": 100,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 0,
        }
        hook._on_model_call(event2)

        # Third call: cache hit
        hook._on_model_call(event2)

        summary = get_current_usage()
        assert summary.call_count == 3
        assert summary.prompt_tokens == 3000
        assert summary.cached_tokens == 1800  # 900 * 2
        assert summary.cache_creation_tokens == 900
        assert summary.cache_hit_count == 2
        assert summary.cache_miss_count == 1
        assert summary.cache_hit_rate == pytest.approx(2 / 3)

    def test_usage_without_cache_fields(self) -> None:
        """Backward compat: usage dict without cache fields still works."""
        hook = CostHook()
        event = MagicMock()
        event.usage = {
            "prompt_tokens": 500,
            "completion_tokens": 100,
        }

        hook._on_model_call(event)
        summary = get_current_usage()

        assert summary.prompt_tokens == 500
        assert summary.completion_tokens == 100
        assert summary.cached_tokens == 0
        assert summary.cache_miss_count == 1  # No cache read = miss
