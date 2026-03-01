"""Real-time inference backend — wraps litellm.acompletion()."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from scout_ai.inference.protocols import InferenceRequest, InferenceResult

log = logging.getLogger(__name__)


class RealTimeBackend:
    """Synchronous real-time inference via litellm.acompletion().

    This is the only built-in backend shipped with Scout AI.
    It extracts the existing litellm call pattern from LLMClient
    into a standalone, protocol-compliant class.
    """

    def __init__(self, *, max_concurrent: int = 8) -> None:
        self._max_concurrent = max_concurrent

    async def infer(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **params: Any,
    ) -> InferenceResult:
        """Single inference call via litellm.acompletion()."""
        from litellm import acompletion

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **params,
        }
        response = await acompletion(**kwargs)
        content = response.choices[0].message.content or ""
        reason = response.choices[0].finish_reason
        mapped_reason = "max_output_reached" if reason == "length" else "finished"

        usage: dict[str, int] = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        return InferenceResult(
            content=content,
            finish_reason=mapped_reason,
            usage=usage,
        )

    async def infer_batch(
        self,
        requests: list[InferenceRequest],
    ) -> list[InferenceResult]:
        """Run multiple inference calls with concurrency control."""
        sem = asyncio.Semaphore(self._max_concurrent)

        async def _bounded(req: InferenceRequest) -> InferenceResult:
            async with sem:
                result = await self.infer(req.messages, req.model, **req.params)
                result.request_id = req.request_id
                return result

        return list(await asyncio.gather(*[_bounded(r) for r in requests]))
