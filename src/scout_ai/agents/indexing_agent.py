"""Indexing agent: builds hierarchical tree indexes from document pages.

Uses Strands Agent with indexing skills to coordinate the pipeline:
detect_toc → process_toc → verify_toc → split_nodes → enrich_nodes → build_index.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from strands import Agent

from scout_ai.agents.factory import create_model
from scout_ai.hooks import AuditHook, CostHook
from scout_ai.prompts.registry import get_prompt
from scout_ai.skills.indexing import (
    build_index,
    detect_toc,
    enrich_nodes,
    fix_incorrect_toc,
    process_toc,
    split_large_nodes,
    verify_toc,
)

if TYPE_CHECKING:
    from scout_ai.core.config import AppSettings

log = logging.getLogger(__name__)


def _resolve_system_prompt(domain: str) -> str:
    """Resolve the indexing system prompt for a given domain.

    Tries the domain-specific prompt first, then falls back to the base prompt.

    Args:
        domain: Domain namespace (e.g. ``"aps"``, ``"workers_comp"``).

    Returns:
        The resolved system prompt string.
    """
    try:
        return get_prompt(domain, "indexing_agent", "INDEXING_SYSTEM_PROMPT")
    except KeyError:
        log.debug("No indexing prompt for domain %r, falling back to base", domain)
        try:
            return get_prompt("base", "indexing_agent", "INDEXING_SYSTEM_PROMPT")
        except KeyError as exc:
            raise KeyError("Base indexing system prompt not found; registry may be misconfigured") from exc


def create_indexing_agent(settings: AppSettings, **kwargs: Any) -> Agent:
    """Create a Strands Agent configured for document indexing.

    Args:
        settings: Application settings (drives model selection + indexing config).
        **kwargs: Additional kwargs passed to Agent constructor.

    Returns:
        Configured Strands Agent with indexing tools.
    """
    model = create_model(settings, model_override=settings.stage_models.indexing_model)
    system_prompt = _resolve_system_prompt(settings.domain)

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            build_index,
            detect_toc,
            process_toc,
            verify_toc,
            fix_incorrect_toc,
            split_large_nodes,
            enrich_nodes,
        ],
        hooks=[AuditHook(), CostHook()],
        trace_attributes={"agent.type": "indexing", "agent.domain": settings.domain},
        name="Scout Indexing Agent",
        description="Builds hierarchical tree indexes from pre-OCR'd document pages",
        **kwargs,
    )
