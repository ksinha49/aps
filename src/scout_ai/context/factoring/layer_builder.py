"""ContextLayerBuilder — constructs multi-breakpoint message hierarchies.

Builds Anthropic/OpenAI-compatible message lists with ``cache_control``
markers at layer boundaries, enabling multi-layer prompt caching:

    system prompt  →  [cache_control]  (stable across all calls)
    document context → [cache_control]  (stable across questions for same doc)
    user query     →  (no cache)        (changes every call)
"""

from __future__ import annotations

from typing import Any

from scout_ai.context.factoring.breakpoint_strategy import compute_breakpoints
from scout_ai.context.models import ContextLayer


class ContextLayerBuilder:
    """Builds multi-layer message structures with cache breakpoints.

    Usage::

        builder = ContextLayerBuilder(max_breakpoints=4)
        messages = builder.build_messages(system_prompt, document_context, query)
        # Pass messages to LLMClient — cache_control is set at layer boundaries.
    """

    def __init__(self, max_breakpoints: int = 4) -> None:
        self._max_breakpoints = max_breakpoints

    def build_layers(
        self,
        system_prompt: str,
        document_context: str,
        query: str,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[ContextLayer]:
        """Build context layers from prompt components.

        Args:
            system_prompt: The system-level instruction prompt.
            document_context: The document text to reason over.
            query: The user's question/extraction prompt.
            tools: Optional tool definitions (rarely used in extraction).

        Returns:
            Ordered list of ContextLayer objects with breakpoints assigned.
        """
        layers: list[ContextLayer] = []

        # Layer 1: System prompt (most stable)
        if system_prompt:
            layers.append(ContextLayer(
                role="system",
                content=[{"type": "text", "text": system_prompt}],
                layer_type="system",
            ))

        # Layer 2: Tools (optional, fairly stable)
        if tools:
            layers.append(ContextLayer(
                role="system",
                content=[{"type": "text", "text": _format_tools(tools)}],
                layer_type="tools",
            ))

        # Layer 3: Document context (changes per document, shared across questions)
        if document_context:
            layers.append(ContextLayer(
                role="system",
                content=[{"type": "text", "text": document_context}],
                layer_type="document",
            ))

        # Layer 4: User query (changes every call — never cached)
        layers.append(ContextLayer(
            role="user",
            content=[{"type": "text", "text": query}],
            layer_type="query",
        ))

        return compute_breakpoints(layers, self._max_breakpoints)

    def build_messages(
        self,
        system_prompt: str,
        document_context: str,
        query: str,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build Anthropic/OpenAI-compatible messages with cache_control.

        Returns a list of message dicts ready for ``LLMClient.complete()``:

        - System layers are merged into a single system message with
          multiple content blocks, each potentially having ``cache_control``.
        - The user query is a standard user message.
        """
        layers = self.build_layers(system_prompt, document_context, query, tools=tools)

        # Merge system layers into one system message
        system_content: list[dict[str, Any]] = []
        user_message: dict[str, Any] | None = None

        for layer in layers:
            if layer.role == "system":
                for block in layer.content:
                    entry = dict(block)
                    if layer.cache_breakpoint:
                        entry["cache_control"] = {"type": "ephemeral"}
                    system_content.append(entry)
            elif layer.role == "user":
                user_message = {"role": "user", "content": layer.content[0]["text"]}

        messages: list[dict[str, Any]] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if user_message:
            messages.append(user_message)

        return messages


def _format_tools(tools: list[dict[str, Any]]) -> str:
    """Format tool definitions as a text block for the system message."""
    import json

    return "Available tools:\n" + json.dumps(tools, indent=2)
