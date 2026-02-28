"""APS classification prompt templates.

Prompts are stored in ``_PROMPT_DATA`` and exposed via ``__getattr__``
which delegates to the prompt registry for DynamoDB / file resolution.
"""

from __future__ import annotations

# ── Raw prompt data (read by FilePromptBackend) ─────────────────────

_PROMPT_DATA: dict[str, str] = {
    "CLASSIFY_SECTION_PROMPT": """
Classify this section of a medical record into one of these types:
face_sheet, progress_note, lab_report, imaging, pathology, operative_report,
discharge_summary, consultation, medication_list, vital_signs, nursing_note,
therapy_note, mental_health, dental, unknown

Section title: {title}
Section content (first 500 chars): {content_preview}

Return JSON:
{{
    "thinking": "<your reasoning>",
    "section_type": "<type from list above>"
}}

Directly return JSON only.""",
}

# ── PEP 562 module __getattr__ ──────────────────────────────────────

_PROMPT_NAMES = frozenset(_PROMPT_DATA.keys())


def __getattr__(name: str) -> str:
    if name in _PROMPT_NAMES:
        from pageindex_rag.prompts.registry import get_prompt

        return get_prompt("aps", "classification", name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(_PROMPT_NAMES) + ["_PROMPT_DATA"]
