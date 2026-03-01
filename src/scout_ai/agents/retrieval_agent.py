"""Retrieval agent: searches document tree indexes for relevant sections.

Uses Strands Agent with retrieval skills (tree_search, batch_retrieve)
to find sections relevant to extraction questions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from strands import Agent

from scout_ai.agents.factory import create_model
from scout_ai.hooks import AuditHook, CostHook
from scout_ai.prompts.templates.base.retrieval_agent import RETRIEVAL_SYSTEM_PROMPT
from scout_ai.skills.retrieval import batch_retrieve, tree_search

if TYPE_CHECKING:
    from scout_ai.core.config import AppSettings


def create_retrieval_agent(settings: AppSettings, **kwargs: Any) -> Agent:
    """Create a Strands Agent configured for document retrieval.

    Args:
        settings: Application settings (drives model selection + retrieval config).
        **kwargs: Additional kwargs passed to Agent constructor.

    Returns:
        Configured Strands Agent with retrieval tools.
    """
    model = create_model(settings)

    return Agent(
        model=model,
        system_prompt=RETRIEVAL_SYSTEM_PROMPT,
        tools=[tree_search, batch_retrieve],
        hooks=[AuditHook(), CostHook()],
        trace_attributes={"agent.type": "retrieval"},
        name="Scout Retrieval Agent",
        description="Searches document tree indexes for relevant sections",
        **kwargs,
    )
