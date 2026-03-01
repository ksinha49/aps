"""Ingestion service: orchestrates index building and persistence."""

from __future__ import annotations

import logging

from scout_ai.interfaces.ingestion import IIngestionProvider
from scout_ai.models import DocumentIndex, PageContent
from scout_ai.services.index_store import IndexStore

log = logging.getLogger(__name__)


class IngestionService:
    """Build + persist document index with dedup check."""

    def __init__(self, provider: IIngestionProvider, store: IndexStore) -> None:
        self._provider = provider
        self._store = store

    async def ingest(
        self,
        pages: list[PageContent],
        doc_id: str,
        doc_name: str,
        force: bool = False,
    ) -> DocumentIndex:
        """Build index and persist. Skips if already exists (unless *force*)."""
        if not force and self._store.exists(doc_id):
            log.info(f"Index already exists for {doc_id}, loading from disk")
            return self._store.load(doc_id)

        index = await self._provider.build_index(pages, doc_id, doc_name)
        self._store.save(index)
        return index
