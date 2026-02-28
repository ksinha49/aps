"""Prompt management: registry, context, and domain-specific templates."""

from __future__ import annotations

from pageindex_rag.prompts.context import PromptContext
from pageindex_rag.prompts.registry import (
    configure,
    get_active_context,
    get_prompt,
    set_active_context,
)

__all__ = [
    "PromptContext",
    "configure",
    "get_active_context",
    "get_prompt",
    "set_active_context",
]
