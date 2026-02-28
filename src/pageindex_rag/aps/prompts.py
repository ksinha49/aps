"""APS-specific prompt templates for indexing, retrieval, and extraction.

This module used to hold prompt constants directly. It now delegates to the
prompt registry via PEP 562 ``__getattr__``, enabling DynamoDB-backed versioned
prompts while maintaining full backward compatibility::

    from pageindex_rag.aps.prompts import TOC_DETECT_PROMPT  # still works
"""

from __future__ import annotations

# ── Mapping: constant name → (domain, category) ────────────────────

_PROMPT_MAP: dict[str, tuple[str, str]] = {
    # Indexing
    "TOC_DETECT_PROMPT": ("aps", "indexing"),
    "GENERATE_TOC_INIT_PROMPT": ("aps", "indexing"),
    "GENERATE_TOC_CONTINUE_PROMPT": ("aps", "indexing"),
    "CHECK_TITLE_APPEARANCE_PROMPT": ("aps", "indexing"),
    "CHECK_TITLE_START_PROMPT": ("aps", "indexing"),
    "GENERATE_SUMMARY_PROMPT": ("aps", "indexing"),
    # Retrieval
    "TREE_SEARCH_PROMPT": ("aps", "retrieval"),
    "CATEGORY_SEARCH_PROMPT": ("aps", "retrieval"),
    # Extraction
    "BATCH_EXTRACTION_PROMPT": ("aps", "extraction"),
    "INDIVIDUAL_EXTRACTION_PROMPT": ("aps", "extraction"),
    # Classification
    "CLASSIFY_SECTION_PROMPT": ("aps", "classification"),
    # Synthesis
    "SYNTHESIS_SYSTEM_PROMPT": ("aps", "synthesis"),
    "SYNTHESIS_PROMPT": ("aps", "synthesis"),
}

_PROMPT_NAMES = frozenset(_PROMPT_MAP.keys())


def __getattr__(name: str) -> str:
    if name in _PROMPT_MAP:
        from pageindex_rag.prompts.registry import get_prompt

        domain, category = _PROMPT_MAP[name]
        return get_prompt(domain, category, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(_PROMPT_NAMES)
