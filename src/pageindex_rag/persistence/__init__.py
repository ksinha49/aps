"""Pluggable persistence backends for document indexes."""

from __future__ import annotations

from pageindex_rag.persistence.file_backend import FilePersistenceBackend
from pageindex_rag.persistence.memory_backend import MemoryPersistenceBackend
from pageindex_rag.persistence.protocols import IPersistenceBackend

__all__ = ["IPersistenceBackend", "FilePersistenceBackend", "MemoryPersistenceBackend"]
