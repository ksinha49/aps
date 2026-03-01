"""JSON file persistence for document indexes."""

from __future__ import annotations

import logging
from pathlib import Path

from scout_ai.models import DocumentIndex

log = logging.getLogger(__name__)


class IndexStore:
    """Save and load ``DocumentIndex`` as JSON files."""

    def __init__(self, store_path: Path) -> None:
        self._path = store_path
        self._path.mkdir(parents=True, exist_ok=True)

    def _index_path(self, doc_id: str) -> Path:
        return self._path / f"{doc_id}.json"

    def exists(self, doc_id: str) -> bool:
        return self._index_path(doc_id).is_file()

    def save(self, index: DocumentIndex) -> Path:
        """Persist index to disk, returning the file path."""
        path = self._index_path(index.doc_id)
        path.write_text(index.model_dump_json(indent=2), encoding="utf-8")
        log.info(f"Saved index to {path}")
        return path

    def load(self, doc_id: str) -> DocumentIndex:
        """Load a previously saved index."""
        path = self._index_path(doc_id)
        if not path.is_file():
            raise FileNotFoundError(f"No index found for doc_id={doc_id} at {path}")
        return DocumentIndex.model_validate_json(path.read_text(encoding="utf-8"))
