"""Inference backend protocol — defines the contract all backends implement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class InferenceRequest:
    """A single inference request to be processed by a backend."""

    request_id: str
    messages: list[dict[str, Any]]
    model: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Result from a single inference call."""

    content: str
    finish_reason: str = "finished"
    usage: dict[str, int] = field(default_factory=dict)
    request_id: str = ""


@runtime_checkable
class IInferenceBackend(Protocol):
    """Protocol for pluggable inference backends.

    Real-time backends resolve immediately; batch/IDP backends resolve
    when the external job completes. Both expose the same async interface.
    """

    async def infer(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **params: Any,
    ) -> InferenceResult:
        """Run a single inference call.

        Args:
            messages: Chat messages in OpenAI format.
            model: Model identifier (supports LiteLLM prefixes).
            **params: Additional parameters (temperature, top_p, etc.).

        Returns:
            InferenceResult with content and metadata.
        """
        ...

    async def infer_batch(
        self,
        requests: list[InferenceRequest],
    ) -> list[InferenceResult]:
        """Run multiple inference calls.

        Args:
            requests: List of InferenceRequest objects.

        Returns:
            List of InferenceResult in the same order as requests.
        """
        ...
