"""Dead letter hook: captures failed tool executions for later analysis."""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strands.hooks.registry import HookRegistry

    from scout_ai.persistence.protocols import IPersistenceBackend

log = logging.getLogger(__name__)


class DeadLetterHook:
    """Strands HookProvider that captures tool failures to a persistence backend.

    Failed tool executions are written as dead-letter entries that can be
    inspected and replayed later.
    """

    def __init__(self, backend: IPersistenceBackend, key_prefix: str = "_dead_letter/") -> None:
        self._backend = backend
        self._key_prefix = key_prefix

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        from strands.hooks.events import AfterToolCallEvent

        registry.add_callback(AfterToolCallEvent, self._on_tool_done)

    def _on_tool_done(self, event: Any) -> None:
        status = getattr(event, "status", None)
        if status == "success":
            return

        tool_name = getattr(event, "tool_name", "unknown")
        error = getattr(event, "error", None)
        invocation_state = getattr(event, "invocation_state", None) or {}
        pipeline_id = invocation_state.get("pipeline_id", "default")

        timestamp = int(time.time() * 1000)
        key = f"{self._key_prefix}{pipeline_id}/{tool_name}/{timestamp}"

        entry = {
            "tool_name": tool_name,
            "status": str(status),
            "error": str(error) if error else None,
            "pipeline_id": pipeline_id,
            "timestamp": timestamp,
        }

        try:
            self._backend.save(key, json.dumps(entry))
            log.warning("Dead letter recorded: %s — %s", key, error)
        except Exception:
            log.error("Failed to write dead letter for %s", tool_name, exc_info=True)

    def list_dead_letters(self, pipeline_id: str = "") -> list[dict[str, Any]]:
        """List all dead letter entries, optionally filtered by pipeline."""
        prefix = f"{self._key_prefix}{pipeline_id}" if pipeline_id else self._key_prefix
        entries = []
        for key in self._backend.list_keys(prefix):
            try:
                entries.append(json.loads(self._backend.load(key)))
            except (KeyError, json.JSONDecodeError):
                continue
        return entries
