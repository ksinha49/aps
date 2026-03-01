"""File-based persistence backend — JSON files on local filesystem."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class FilePersistenceBackend:
    """Stores data as JSON files in a local directory."""

    def __init__(self, base_path: Path) -> None:
        self._base = base_path
        self._base.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        safe_key = key.replace("/", "_").replace("\\", "_")
        if not safe_key.endswith(".json"):
            safe_key += ".json"
        return self._base / safe_key

    def save(self, key: str, data: str) -> None:
        path = self._key_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
        log.debug(f"Saved {key} to {path}")

    def load(self, key: str) -> str:
        path = self._key_path(key)
        if not path.is_file():
            raise KeyError(f"Not found: {key} (path: {path})")
        return path.read_text(encoding="utf-8")

    def exists(self, key: str) -> bool:
        return self._key_path(key).is_file()

    def delete(self, key: str) -> None:
        path = self._key_path(key)
        if path.is_file():
            path.unlink()

    def list_keys(self, prefix: str = "") -> list[str]:
        keys = []
        for path in self._base.glob("*.json"):
            key = path.stem
            if key.startswith(prefix):
                keys.append(key)
        return sorted(keys)
