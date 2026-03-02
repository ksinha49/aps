"""Cache key computation for deterministic, collision-resistant keys."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scout_ai.models import DocumentIndex


def compute_cache_key(
    question_id: str,
    index_hash: str,
    model_name: str,
    context_hash: str = "",
) -> str:
    """Compute a deterministic SHA-256 cache key.

    The key incorporates question identity, document state, model version,
    and optionally the context hash to ensure cache correctness.
    """
    parts = [question_id, index_hash, model_name]
    if context_hash:
        parts.append(context_hash)
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


def compute_index_hash(index: DocumentIndex) -> str:
    """Compute a stable hash from document index metadata.

    Uses doc_id, total_pages, and tree_count to produce a hash that
    changes when the document is re-indexed but stays stable otherwise.
    """
    data: dict[str, Any] = {
        "doc_id": index.doc_id,
        "total_pages": index.total_pages,
        "tree_count": len(index.trees) if index.trees else 0,
    }
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
