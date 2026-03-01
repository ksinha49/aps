"""Retrieval agent: searches document tree indexes for relevant sections.

Uses Strands Agent with retrieval skills (tree_search, batch_retrieve)
to find sections relevant to extraction questions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from strands import Agent

from scout_ai.agents.factory import create_model
from scout_ai.hooks import AuditHook, CostHook
from scout_ai.prompts.registry import get_prompt
from scout_ai.skills.retrieval import batch_retrieve, tree_search

if TYPE_CHECKING:
    from scout_ai.core.config import AppSettings

log = logging.getLogger(__name__)


def _resolve_system_prompt(domain: str) -> str:
    """Resolve the retrieval system prompt for a given domain.

    Tries the domain-specific prompt first, then falls back to the base prompt.

    Args:
        domain: Domain namespace (e.g. ``"aps"``, ``"workers_comp"``).

    Returns:
        The resolved system prompt string.
    """
    try:
        return get_prompt(domain, "retrieval_agent", "RETRIEVAL_SYSTEM_PROMPT")
    except KeyError:
        log.debug("No retrieval prompt for domain %r, falling back to base", domain)
        try:
            return get_prompt("base", "retrieval_agent", "RETRIEVAL_SYSTEM_PROMPT")
        except KeyError as exc:
            raise KeyError("Base retrieval system prompt not found; registry may be misconfigured") from exc


def create_retrieval_agent(settings: AppSettings, **kwargs: Any) -> Agent:
    """Create a Strands Agent configured for document retrieval.

    Args:
        settings: Application settings (drives model selection + retrieval config).
        **kwargs: Additional kwargs passed to Agent constructor.

    Returns:
        Configured Strands Agent with retrieval tools.
    """
    model = create_model(settings, model_override=settings.stage_models.retrieval_model)
    system_prompt = _resolve_system_prompt(settings.domain)

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[tree_search, batch_retrieve],
        hooks=[AuditHook(), CostHook()],
        trace_attributes={"agent.type": "retrieval", "agent.domain": settings.domain},
        name="Scout Retrieval Agent",
        description="Searches document tree indexes for relevant sections",
        **kwargs,
    )
