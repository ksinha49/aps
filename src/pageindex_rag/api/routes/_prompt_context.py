"""Shared prompt context helpers for API routes."""

from __future__ import annotations

from pydantic import BaseModel

from pageindex_rag.core.config import PromptConfig
from pageindex_rag.prompts.context import PromptContext
from pageindex_rag.prompts.registry import set_active_context


class PromptContextOverride(BaseModel):
    """Optional per-request prompt dimension overrides."""

    lob: str | None = None
    department: str | None = None
    use_case: str | None = None
    process: str | None = None


def resolve_context(override: PromptContextOverride | None, defaults: PromptConfig) -> PromptContext:
    """Merge per-request overrides onto global defaults."""
    return PromptContext(
        lob=override.lob if override and override.lob else defaults.default_lob,
        department=override.department if override and override.department else defaults.default_department,
        use_case=override.use_case if override and override.use_case else defaults.default_use_case,
        process=override.process if override and override.process else defaults.default_process,
    )


def apply_prompt_context(override: PromptContextOverride | None, defaults: PromptConfig) -> None:
    """Resolve and set the per-request prompt context ContextVar."""
    ctx = resolve_context(override, defaults)
    set_active_context(ctx)
