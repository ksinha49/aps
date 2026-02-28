"""Deterministic Strands model for testing — no real LLM calls.

Usage::

    model = FakeStrandsModel(responses=[
        '{"node_ids": ["001"], "reasoning": "test"}',
        '{"answer": "42", "confidence": 0.9}',
    ])
    agent = Agent(model=model, tools=[...])
    result = agent("Find the answer")

    assert model.calls[0]["messages"]  # inspect what was sent
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterable


@dataclass
class FakeModelCall:
    """Record of a single call to the fake model."""

    messages: list[dict[str, Any]]
    tool_specs: list[dict[str, Any]] | None
    system_prompt: str | None


class FakeStrandsModel:
    """A Strands Model implementation that returns canned responses.

    Implements the minimal Model interface needed for Agent testing:
    ``update_config``, ``get_config``, and ``stream``.

    Args:
        responses: Ordered list of response strings. Each ``stream()`` call
            consumes the next response. If exhausted, returns a default.
        default_response: Fallback when ``responses`` is exhausted.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        default_response: str = '{"status": "ok"}',
    ) -> None:
        self._responses: list[str] = list(responses or [])
        self._default = default_response
        self._call_index = 0
        self.calls: list[FakeModelCall] = []
        self._config: dict[str, Any] = {}

    def update_config(self, **model_config: Any) -> None:
        self._config.update(model_config)

    def get_config(self) -> dict[str, Any]:
        return dict(self._config)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[dict[str, Any]]:
        """Yield streaming events that mimic a real model response.

        The response is returned as a sequence of:
        1. ``messageStart`` event
        2. ``contentBlockStart`` event
        3. ``contentBlockDelta`` events (one per character chunk)
        4. ``contentBlockStop`` event
        5. ``messageStop`` event with usage metadata
        """
        self.calls.append(FakeModelCall(
            messages=messages,
            tool_specs=tool_specs,
            system_prompt=system_prompt,
        ))

        if self._call_index < len(self._responses):
            text = self._responses[self._call_index]
        else:
            text = self._default
        self._call_index += 1

        async def _generate() -> AsyncIterable[dict[str, Any]]:
            yield {
                "messageStart": {
                    "role": "assistant",
                },
            }
            yield {
                "contentBlockStart": {
                    "start": {"text": ""},
                },
            }
            # Emit text in chunks
            chunk_size = max(1, len(text) // 3) if text else 1
            for i in range(0, max(1, len(text)), chunk_size):
                chunk = text[i : i + chunk_size]
                yield {
                    "contentBlockDelta": {
                        "delta": {"text": chunk},
                    },
                }
            yield {
                "contentBlockStop": {},
            }
            yield {
                "messageStop": {
                    "stopReason": "end_turn",
                },
            }
            yield {
                "metadata": {
                    "usage": {
                        "inputTokens": len(str(messages)) // 4,
                        "outputTokens": len(text) // 4,
                    },
                    "metrics": {
                        "latencyMs": 10,
                    },
                },
            }

        return _generate()

    def structured_output(self, output_model: Any, prompt: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("FakeStrandsModel does not support structured_output")
