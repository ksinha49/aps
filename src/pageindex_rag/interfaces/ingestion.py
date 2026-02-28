"""Abstract ingestion (index building) provider."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pageindex_rag.models import DocumentIndex, PageContent


class IIngestionProvider(ABC):
    @abstractmethod
    async def build_index(
        self,
        pages: list[PageContent],
        doc_id: str,
        doc_name: str,
    ) -> DocumentIndex:
        """Build a hierarchical tree index from pre-OCR'd pages."""
