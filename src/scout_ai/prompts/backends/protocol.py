"""Protocol for pluggable prompt backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from scout_ai.prompts.context import PromptContext


@runtime_checkable
class IPromptBackend(Protocol):
    """Interface for prompt storage backends.

    Implementations must be synchronous — prompt resolution happens inline
    during module ``__getattr__`` calls which cannot be async.
    """

    def get(
        self,
        domain: str,
        category: str,
        name: str,
        *,
        context: PromptContext | None = None,
        version: int | None = None,
    ) -> str:
        """Retrieve a prompt template.

        Args:
            domain: Domain namespace (e.g. ``"aps"``).
            category: Prompt category (e.g. ``"indexing"``).
            name: Constant name (e.g. ``"TOC_DETECT_PROMPT"``).
            context: Optional business dimensions for resolution.
            version: Optional specific version number.

        Returns:
            The prompt template string.

        Raises:
            KeyError: If the prompt is not found.
        """
        ...
