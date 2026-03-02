"""PrefixStabilizer — deterministic context ordering for prompt cache hits."""

from __future__ import annotations

import json
from typing import Any

from scout_ai.context.prefix.sort_strategies import (
    sort_by_doc_id_page,
    sort_by_page_number,
    sort_by_section_path,
)

_STRATEGIES = {
    "page_number": sort_by_page_number,
    "section_path": sort_by_section_path,
    "doc_id_page": sort_by_doc_id_page,
}


class PrefixStabilizer:
    """Applies a sort strategy to retrieved nodes for deterministic ordering.

    Deterministic ordering ensures that the same set of retrieved nodes
    always produces the same context string, which maximizes prompt
    cache hit rates across extraction calls.
    """

    def __init__(self, strategy: str = "page_number") -> None:
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"Unknown sort strategy {strategy!r}. "
                f"Choose from: {', '.join(sorted(_STRATEGIES))}"
            )
        self._strategy = _STRATEGIES[strategy]
        self._strategy_name = strategy

    def stabilize(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a deterministically sorted copy of the node list."""
        return self._strategy(nodes)

    @staticmethod
    def stabilize_json(data: Any) -> str:
        """Serialize data to deterministic JSON (sorted keys, no extra whitespace)."""
        return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
