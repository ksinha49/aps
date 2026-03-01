"""Prompt registry: configurable backend routing with file/DynamoDB support.

Usage::

    # Default (file backend, auto-configured on first call):
    prompt = get_prompt("aps", "indexing", "TOC_DETECT_PROMPT")

    # Explicit configuration:
    from scout_ai.prompts import configure, PromptContext
    configure(backend="dynamodb", table_name="my-table", aws_region="us-east-1")

    # With business-dimension context:
    prompt = get_prompt("aps", "indexing", "TOC_DETECT_PROMPT",
                        context=PromptContext(lob="life", department="uw"))

    # Per-request context (set by API middleware):
    from scout_ai.prompts import set_active_context
    set_active_context(PromptContext(lob="life"))
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Literal

from scout_ai.prompts.backends.file_backend import FilePromptBackend
from scout_ai.prompts.backends.protocol import IPromptBackend
from scout_ai.prompts.context import PromptContext

logger = logging.getLogger(__name__)

# ── Module-level state ──────────────────────────────────────────────

_primary_backend: IPromptBackend | None = None
_fallback_backend: IPromptBackend | None = None
_default_context: PromptContext = PromptContext()
_configured: bool = False

_active_context: ContextVar[PromptContext | None] = ContextVar("_active_context", default=None)


# ── Public API ──────────────────────────────────────────────────────


def configure(
    *,
    backend: Literal["file", "dynamodb"] = "file",
    table_name: str = "scout-prompts",
    aws_region: str = "us-east-1",
    cache_ttl_seconds: float = 300.0,
    cache_max_size: int = 500,
    fallback_to_file: bool = True,
    default_lob: str = "*",
    default_department: str = "*",
    default_use_case: str = "*",
    default_process: str = "*",
) -> None:
    """Initialize the prompt registry with the chosen backend.

    Should be called once at application startup (e.g. in FastAPI lifespan).
    If never called, the first ``get_prompt()`` call auto-configures to ``"file"``.
    """
    global _primary_backend, _fallback_backend, _default_context, _configured

    _default_context = PromptContext(
        lob=default_lob,
        department=default_department,
        use_case=default_use_case,
        process=default_process,
    )

    if backend == "dynamodb":
        from scout_ai.prompts.backends.dynamodb_backend import DynamoDBPromptBackend

        _primary_backend = DynamoDBPromptBackend(
            table_name=table_name,
            aws_region=aws_region,
            cache_ttl_seconds=cache_ttl_seconds,
            cache_max_size=cache_max_size,
        )
        _fallback_backend = FilePromptBackend() if fallback_to_file else None
    else:
        _primary_backend = FilePromptBackend()
        _fallback_backend = None

    _configured = True


def get_prompt(
    domain: str,
    category: str,
    name: str,
    *,
    context: PromptContext | None = None,
    version: int | None = None,
) -> str:
    """Look up a prompt template by domain, category, and name.

    Context resolution order:
    1. Explicit ``context`` argument (if provided)
    2. Per-request ``ContextVar`` (set by ``set_active_context()``)
    3. Global ``_default_context`` (set by ``configure()``)

    Backend chain:
    1. Primary backend (file or DynamoDB)
    2. Fallback backend (file, if ``fallback_to_file=True``)
    3. ``KeyError`` if still not found

    Args:
        domain: Domain namespace (e.g. ``"aps"``).
        category: Prompt category (e.g. ``"indexing"``, ``"retrieval"``).
        name: Constant name (e.g. ``"TOC_DETECT_PROMPT"``).
        context: Optional business dimensions for resolution.
        version: Optional specific version number.

    Returns:
        The prompt template string.

    Raises:
        KeyError: If the prompt is not found in any backend.
    """
    _ensure_configured()

    assert _primary_backend is not None  # guaranteed by _ensure_configured

    # Resolve context
    resolved_ctx = context or _active_context.get() or _default_context

    # Try primary backend
    try:
        return _primary_backend.get(domain, category, name, context=resolved_ctx, version=version)
    except KeyError:
        if _fallback_backend is not None:
            logger.debug("Primary backend miss for %s/%s/%s, trying fallback", domain, category, name)
            return _fallback_backend.get(domain, category, name, context=resolved_ctx, version=version)
        raise


def set_active_context(ctx: PromptContext) -> None:
    """Set the per-request prompt context (call from API middleware/route handlers)."""
    _active_context.set(ctx)


def get_active_context() -> PromptContext | None:
    """Get the current per-request prompt context, if any."""
    return _active_context.get()


def reset() -> None:
    """Reset the registry to unconfigured state (for testing)."""
    global _primary_backend, _fallback_backend, _default_context, _configured
    _primary_backend = None
    _fallback_backend = None
    _default_context = PromptContext()
    _configured = False
    _active_context.set(None)


# ── Internal ────────────────────────────────────────────────────────


def _ensure_configured() -> None:
    """Auto-configure to file backend if ``configure()`` was never called."""
    global _configured
    if not _configured:
        configure(backend="file")
