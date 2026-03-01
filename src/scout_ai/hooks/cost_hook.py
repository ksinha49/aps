"""Cost tracking hook: accumulates token usage per request, including cache metrics."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strands.hooks.registry import HookRegistry


@dataclass
class UsageSummary:
    """Accumulated token usage for a single request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    call_count: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of calls that hit the prompt cache."""
        total = self.cache_hit_count + self.cache_miss_count
        return self.cache_hit_count / total if total > 0 else 0.0

    @property
    def estimated_savings_ratio(self) -> float:
        """Ratio of tokens saved via cache reads (0.9x savings per cached token)."""
        if self.prompt_tokens == 0:
            return 0.0
        return (self.cached_tokens * 0.9) / self.prompt_tokens


_usage: ContextVar[UsageSummary] = ContextVar("scout_usage")


def get_current_usage() -> UsageSummary:
    """Get the token usage for the current request context."""
    try:
        return _usage.get()
    except LookupError:
        summary = UsageSummary()
        _usage.set(summary)
        return summary


def reset_usage() -> UsageSummary:
    """Reset and return a fresh usage tracker for the current context."""
    summary = UsageSummary()
    _usage.set(summary)
    return summary


class CostHook:
    """Strands HookProvider that tracks token usage per request.

    Use ``get_current_usage()`` to read accumulated totals after an agent run.
    Tracks Anthropic prompt caching metrics (cache reads/writes) in addition
    to standard prompt/completion token counts.
    """

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        from strands.hooks.events import AfterModelCallEvent

        registry.add_callback(AfterModelCallEvent, self._on_model_call)

    def _on_model_call(self, event: Any) -> None:
        usage_data = getattr(event, "usage", {}) or {}
        summary = get_current_usage()
        summary.prompt_tokens += usage_data.get("inputTokens", usage_data.get("prompt_tokens", 0))
        summary.completion_tokens += usage_data.get("outputTokens", usage_data.get("completion_tokens", 0))
        summary.cached_tokens += usage_data.get("cache_read_input_tokens", 0)
        summary.cache_creation_tokens += usage_data.get("cache_creation_input_tokens", 0)
        summary.call_count += 1
        if usage_data.get("cache_read_input_tokens", 0) > 0:
            summary.cache_hit_count += 1
        else:
            summary.cache_miss_count += 1
