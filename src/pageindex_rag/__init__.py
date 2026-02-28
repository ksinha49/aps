"""pageindex-rag: Vectorless RAG with hierarchical tree indexes, powered by Strands Agents SDK.

Legacy API (unchanged)::

    from pageindex_rag import (
        PageIndexSettings,
        PageContent, DocumentIndex, TreeNode, Citation,
        ExtractionQuestion, ExtractionResult, BatchExtractionResult,
        PageIndexIndexer, PageIndexRetrieval, PageIndexChat,
        IngestionService, ExtractionService, IndexStore,
        LLMClient,
    )

Strands-era API::

    from pageindex_rag import (
        AppSettings,
        create_indexing_agent, create_retrieval_agent, create_extraction_agent,
        AuditHook, CostHook, CheckpointHook, CircuitBreakerHook, DeadLetterHook,
    )
"""

from __future__ import annotations

# ── Legacy imports (backward compatible) ──────────────────────────────

from pageindex_rag.config import PageIndexSettings
from pageindex_rag.models import (
    BatchExtractionResult,
    Citation,
    DocumentIndex,
    ExtractionCategory,
    ExtractionQuestion,
    ExtractionResult,
    MedicalSectionType,
    PageContent,
    RetrievalResult,
    TreeNode,
)
from pageindex_rag.providers.pageindex.chat import PageIndexChat
from pageindex_rag.providers.pageindex.client import LLMClient
from pageindex_rag.providers.pageindex.indexer import PageIndexIndexer
from pageindex_rag.providers.pageindex.retrieval import PageIndexRetrieval
from pageindex_rag.services.extraction_service import ExtractionService
from pageindex_rag.services.index_store import IndexStore
from pageindex_rag.services.ingestion_service import IngestionService

# ── Strands-era imports ───────────────────────────────────────────────

from pageindex_rag.agents.extraction_agent import create_extraction_agent
from pageindex_rag.agents.indexing_agent import create_indexing_agent
from pageindex_rag.agents.retrieval_agent import create_retrieval_agent
from pageindex_rag.core.config import AppSettings
from pageindex_rag.hooks import (
    AuditHook,
    CheckpointHook,
    CircuitBreakerHook,
    CostHook,
    DeadLetterHook,
)

__all__ = [
    # Legacy
    "PageIndexSettings",
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
    "PageIndexIndexer",
    "PageIndexRetrieval",
    "PageIndexChat",
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
