"""Checkpoint hook: saves pipeline state after each successful tool for resume."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pageindex_rag.persistence.protocols import IPersistenceBackend
    from strands.hooks.registry import HookRegistry

log = logging.getLogger(__name__)


class CheckpointHook:
    """Strands HookProvider that checkpoints state after each tool success.

    Enables resuming a pipeline from the last successful step after a crash.
    """

    def __init__(self, backend: IPersistenceBackend) -> None:
        self._backend = backend

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        from strands.hooks.events import AfterToolCallEvent

        registry.add_callback(AfterToolCallEvent, self._on_tool_done)

    def _on_tool_done(self, event: Any) -> None:
        status = getattr(event, "status", None)
        if status != "success":
            return

        invocation_state = getattr(event, "invocation_state", None) or {}
        pipeline_id = invocation_state.get("pipeline_id", "default")
        tool_name = getattr(event, "tool_name", "unknown")

        key = f"_checkpoint/{pipeline_id}/{tool_name}"
        result = getattr(event, "result", None)

        try:
            payload = json.dumps({"tool": tool_name, "result": str(result)})
            self._backend.save(key, payload)
            log.debug("Checkpoint saved: %s", key)
        except Exception:
            log.warning("Failed to save checkpoint for %s", key, exc_info=True)

    def load_checkpoint(self, pipeline_id: str, tool_name: str) -> dict[str, Any] | None:
        """Load a checkpoint for a specific pipeline step."""
        key = f"_checkpoint/{pipeline_id}/{tool_name}"
        if not self._backend.exists(key):
            return None
        try:
            return json.loads(self._backend.load(key))
        except (KeyError, json.JSONDecodeError):
            return None

    def clear_checkpoints(self, pipeline_id: str) -> None:
        """Remove all checkpoints for a pipeline run."""
        prefix = f"_checkpoint/{pipeline_id}/"
        for key in self._backend.list_keys(prefix):
            self._backend.delete(key)
