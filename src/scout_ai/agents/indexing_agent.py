"""Indexing agent: builds hierarchical tree indexes from document pages.

Uses Strands Agent with indexing skills to coordinate the pipeline:
detect_toc → process_toc → verify_toc → split_nodes → enrich_nodes → build_index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from strands import Agent

from scout_ai.agents.factory import create_model
from scout_ai.hooks import AuditHook, CostHook
from scout_ai.prompts.templates.base.indexing_agent import INDEXING_SYSTEM_PROMPT
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


def create_indexing_agent(settings: AppSettings, **kwargs: Any) -> Agent:
    """Create a Strands Agent configured for document indexing.

    Args:
        settings: Application settings (drives model selection + indexing config).
        **kwargs: Additional kwargs passed to Agent constructor.

    Returns:
        Configured Strands Agent with indexing tools.
    """
    model = create_model(settings)

    return Agent(
        model=model,
        system_prompt=INDEXING_SYSTEM_PROMPT,
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
        trace_attributes={"agent.type": "indexing"},
        name="Scout Indexing Agent",
        description="Builds hierarchical tree indexes from pre-OCR'd document pages",
        **kwargs,
    )
