"""Audit hook: logs every LLM call and tool execution for traceability."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strands.hooks.registry import HookRegistry

log = logging.getLogger(__name__)


class AuditHook:
    """Strands HookProvider that logs model calls and tool executions.

    Registers callbacks for ``AfterModelCallEvent`` and ``AfterToolCallEvent``
    to create an audit trail of all agent activity.
    """

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        from strands.hooks.events import AfterModelCallEvent, AfterToolCallEvent

        registry.add_callback(AfterModelCallEvent, self._on_model_call)
        registry.add_callback(AfterToolCallEvent, self._on_tool_call)

    def _on_model_call(self, event: Any) -> None:
        usage = getattr(event, "usage", {}) or {}
        log.info(
            "llm_call | model=%s prompt_tokens=%s completion_tokens=%s "
            "cached_tokens=%s cache_creation=%s latency_ms=%s",
            getattr(event, "model_id", "unknown"),
            usage.get("inputTokens", usage.get("prompt_tokens", "?")),
            usage.get("outputTokens", usage.get("completion_tokens", "?")),
            usage.get("cache_read_input_tokens", 0),
            usage.get("cache_creation_input_tokens", 0),
            getattr(event, "latency_ms", "?"),
        )

    def _on_tool_call(self, event: Any) -> None:
        log.info(
            "tool_call | tool=%s status=%s latency_ms=%s",
            getattr(event, "tool_name", "unknown"),
            getattr(event, "status", "unknown"),
            getattr(event, "latency_ms", "?"),
        )
