"""Fake inference backend for testing."""

from __future__ import annotations

from typing import Any

from scout_ai.inference.protocols import InferenceRequest, InferenceResult


class FakeInferenceBackend:
    """Canned-response inference backend for tests — no LLM calls needed."""

    def __init__(
        self,
        *,
        default_content: str = "fake response",
        default_finish_reason: str = "finished",
    ) -> None:
        self._default_content = default_content
        self._default_finish_reason = default_finish_reason
        self.calls: list[dict[str, Any]] = []

    async def infer(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **params: Any,
    ) -> InferenceResult:
        self.calls.append({"messages": messages, "model": model, "params": params})
        return InferenceResult(
            content=self._default_content,
            finish_reason=self._default_finish_reason,
        )

    async def infer_batch(
        self,
        requests: list[InferenceRequest],
    ) -> list[InferenceResult]:
        results: list[InferenceResult] = []
        for req in requests:
            result = await self.infer(req.messages, req.model, **req.params)
            result.request_id = req.request_id
            results.append(result)
        return results
