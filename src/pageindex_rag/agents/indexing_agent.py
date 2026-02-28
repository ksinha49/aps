"""Indexing agent: builds hierarchical tree indexes from document pages.

Uses Strands Agent with indexing skills to coordinate the pipeline:
detect_toc → process_toc → verify_toc → split_nodes → enrich_nodes → build_index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from strands import Agent

from pageindex_rag.agents.factory import create_model
from pageindex_rag.hooks import AuditHook, CostHook
from pageindex_rag.prompts.templates.base.indexing_agent import INDEXING_SYSTEM_PROMPT
from pageindex_rag.skills.indexing import (
    build_index,
    detect_toc,
    enrich_nodes,
    fix_incorrect_toc,
    process_toc,
    split_large_nodes,
    verify_toc,
)

if TYPE_CHECKING:
    from pageindex_rag.core.config import AppSettings


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
        name="PageIndex Indexing Agent",
        description="Builds hierarchical tree indexes from pre-OCR'd document pages",
        **kwargs,
    )
