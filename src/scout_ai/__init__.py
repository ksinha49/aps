"""scout-ai: Vectorless RAG with hierarchical tree indexes, powered by Strands Agents SDK.

Legacy API (unchanged)::

    from scout_ai import (
        ScoutSettings,
        PageContent, DocumentIndex, TreeNode, Citation,
        ExtractionQuestion, ExtractionResult, BatchExtractionResult,
        ScoutIndexer, ScoutRetrieval, ScoutChat,
        IngestionService, ExtractionService, IndexStore,
        LLMClient,
    )

Strands-era API::

    from scout_ai import (
        AppSettings,
        create_indexing_agent, create_retrieval_agent, create_extraction_agent,
        AuditHook, CostHook, CheckpointHook, CircuitBreakerHook, DeadLetterHook,
    )
"""

from __future__ import annotations

from typing import Any

# ── Strands-era imports ───────────────────────────────────────────────
from scout_ai.agents.extraction_agent import create_extraction_agent
from scout_ai.agents.indexing_agent import create_indexing_agent
from scout_ai.agents.retrieval_agent import create_retrieval_agent

# ── Legacy imports (backward compatible) ──────────────────────────────
from scout_ai.config import ScoutSettings
from scout_ai.core.config import AppSettings
from scout_ai.hooks import (
    AuditHook,
    CheckpointHook,
    CircuitBreakerHook,
    CostHook,
    DeadLetterHook,
)
from scout_ai.models import (
    BatchExtractionResult,
    Citation,
    DocumentIndex,
    ExtractionQuestion,
    ExtractionResult,
    PageContent,
    RetrievalResult,
    TreeNode,
)
from scout_ai.providers.pageindex.chat import ScoutChat
from scout_ai.providers.pageindex.client import LLMClient
from scout_ai.providers.pageindex.indexer import ScoutIndexer
from scout_ai.providers.pageindex.retrieval import ScoutRetrieval
from scout_ai.services.extraction_service import ExtractionService
from scout_ai.services.index_store import IndexStore
from scout_ai.services.ingestion_service import IngestionService

__all__ = [
    # Legacy
    "ScoutSettings",
    "PageContent",
    "TreeNode",
    "DocumentIndex",
    "MedicalSectionType",
    "ExtractionCategory",
    "ExtractionQuestion",
    "ExtractionResult",
    "Citation",
    "RetrievalResult",
    "BatchExtractionResult",
    "LLMClient",
    "ScoutIndexer",
    "ScoutRetrieval",
    "ScoutChat",
    "IngestionService",
    "ExtractionService",
    "IndexStore",
    # Strands-era
    "AppSettings",
    "create_indexing_agent",
    "create_retrieval_agent",
    "create_extraction_agent",
    "AuditHook",
    "CostHook",
    "CheckpointHook",
    "CircuitBreakerHook",
    "DeadLetterHook",
]


# Backward compat: ``from scout_ai import MedicalSectionType`` still works
_COMPAT_NAMES = {"MedicalSectionType", "ExtractionCategory"}


def __getattr__(name: str) -> Any:
    if name in _COMPAT_NAMES:
        from scout_ai.domains.aps.models import (
            ExtractionCategory as _EC,
        )
        from scout_ai.domains.aps.models import (
            MedicalSectionType as _MST,
        )

        _map = {"MedicalSectionType": _MST, "ExtractionCategory": _EC}
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
