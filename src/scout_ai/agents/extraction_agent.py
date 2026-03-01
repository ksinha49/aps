"""Extraction agent: extracts precise answers from medical document context.

Uses Strands Agent with extraction skills (extract_batch, extract_individual)
for tiered answer extraction with citation grounding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from strands import Agent

from scout_ai.agents.factory import create_model
from scout_ai.hooks import AuditHook, CostHook
from scout_ai.prompts.templates.base.extraction_agent import EXTRACTION_SYSTEM_PROMPT
from scout_ai.skills.extraction import extract_batch, extract_individual

if TYPE_CHECKING:
    from scout_ai.core.config import AppSettings


def create_extraction_agent(settings: AppSettings, **kwargs: Any) -> Agent:
    """Create a Strands Agent configured for answer extraction.

    Args:
        settings: Application settings (drives model selection + extraction config).
        **kwargs: Additional kwargs passed to Agent constructor.

    Returns:
        Configured Strands Agent with extraction tools.
    """
    model = create_model(settings)

    return Agent(
        model=model,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        tools=[extract_batch, extract_individual],
        hooks=[AuditHook(), CostHook()],
        trace_attributes={"agent.type": "extraction"},
        name="Scout Extraction Agent",
        description="Extracts precise answers from medical document context with citations",
        **kwargs,
    )
