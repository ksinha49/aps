"""Pluggable persistence backends for document indexes."""

from __future__ import annotations

from scout_ai.persistence.file_backend import FilePersistenceBackend
from scout_ai.persistence.memory_backend import MemoryPersistenceBackend
from scout_ai.persistence.protocols import IPersistenceBackend

__all__ = ["IPersistenceBackend", "FilePersistenceBackend", "MemoryPersistenceBackend"]
