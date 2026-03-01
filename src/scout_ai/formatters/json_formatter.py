"""JSON output formatter — simple companion for testing and API responses."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from scout_ai.synthesis.models import UnderwriterSummary


class JSONFormatter:
    """Renders UnderwriterSummary as indented JSON bytes."""

    def format(self, summary: UnderwriterSummary, **kwargs: Any) -> bytes:
        """Serialize *summary* to pretty-printed JSON bytes."""
        return json.dumps(dataclasses.asdict(summary), indent=2, default=str).encode()

    def format_to_file(self, summary: UnderwriterSummary, path: Path, **kwargs: Any) -> Path:
        """Write JSON to *path* and return it."""
        path.write_bytes(self.format(summary, **kwargs))
        return path

    @property
    def content_type(self) -> str:
        return "application/json"
