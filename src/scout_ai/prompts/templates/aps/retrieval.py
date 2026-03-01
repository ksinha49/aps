"""APS retrieval prompt templates.

Prompts are stored in ``_PROMPT_DATA`` and exposed via ``__getattr__``
which delegates to the prompt registry for DynamoDB / file resolution.
"""

from __future__ import annotations

# ── Raw prompt data (read by FilePromptBackend) ─────────────────────

_PROMPT_DATA: dict[str, str] = {
    "TREE_SEARCH_PROMPT": """
You are searching through the structure of an Attending Physician Statement (APS) to find
sections relevant to the query. The document tree shows medical record sections with their
page ranges and summaries.

Document structure:
{tree_structure}

Query: {query}

Which sections (by node_id) are most likely to contain information relevant to this query?
Consider: progress notes often contain vital signs and assessments, lab reports contain test
results, imaging reports contain scan findings, discharge summaries contain treatment outcomes.

Return JSON:
{{
    "reasoning": "<why these sections are relevant>",
    "node_ids": ["<node_id_1>", "<node_id_2>", ...]
}}

Return up to {top_k} most relevant node_ids. Directly return JSON only.""",
    "CATEGORY_SEARCH_PROMPT": """
You are searching an APS (Attending Physician Statement) medical record tree to find sections
relevant to the following extraction category.

Category: {category}
Category description: {category_description}

Document structure:
{tree_structure}

Which sections (by node_id) are most likely to contain {category} information?

Return JSON:
{{
    "reasoning": "<why these sections>",
    "node_ids": ["<node_id_1>", "<node_id_2>", ...]
}}

Return up to {top_k} most relevant node_ids. Directly return JSON only.""",
}

# ── PEP 562 module __getattr__ ──────────────────────────────────────

_PROMPT_NAMES = frozenset(_PROMPT_DATA.keys())


def __getattr__(name: str) -> str:
    if name in _PROMPT_NAMES:
        from scout_ai.prompts.registry import get_prompt

        return get_prompt("aps", "retrieval", name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(_PROMPT_NAMES) + ["_PROMPT_DATA"]
