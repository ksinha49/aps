"""Extraction agent: extracts precise answers from medical document context.

Uses Strands Agent with extraction skills (extract_batch, extract_individual)
for tiered answer extraction with citation grounding.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from strands import Agent

from scout_ai.agents.factory import create_model
from scout_ai.hooks import AuditHook, CostHook
from scout_ai.prompts.registry import get_prompt
from scout_ai.skills.extraction import extract_batch, extract_individual

if TYPE_CHECKING:
    from scout_ai.core.config import AppSettings

log = logging.getLogger(__name__)


def _resolve_system_prompt(domain: str) -> str:
    """Resolve the extraction system prompt for a given domain.

    Tries the domain-specific prompt first, then falls back to the base prompt.

    Args:
        domain: Domain namespace (e.g. ``"aps"``, ``"workers_comp"``).

    Returns:
        The resolved system prompt string.
    """
    try:
        return get_prompt(domain, "extraction_agent", "EXTRACTION_SYSTEM_PROMPT")
    except KeyError:
        log.debug("No extraction prompt for domain %r, falling back to base", domain)
        try:
            return get_prompt("base", "extraction_agent", "EXTRACTION_SYSTEM_PROMPT")
        except KeyError as exc:
            raise KeyError("Base extraction system prompt not found; registry may be misconfigured") from exc


def create_extraction_agent(settings: AppSettings, **kwargs: Any) -> Agent:
    """Create a Strands Agent configured for answer extraction.

    Args:
        settings: Application settings (drives model selection + extraction config).
        **kwargs: Additional kwargs passed to Agent constructor.

    Returns:
        Configured Strands Agent with extraction tools.
    """
    model = create_model(settings, model_override=settings.stage_models.extraction_model)
    system_prompt = _resolve_system_prompt(settings.domain)

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[extract_batch, extract_individual],
        hooks=[AuditHook(), CostHook()],
        trace_attributes={"agent.type": "extraction", "agent.domain": settings.domain},
        name="Scout Extraction Agent",
        description="Extracts precise answers from medical document context with citations",
        **kwargs,
    )
