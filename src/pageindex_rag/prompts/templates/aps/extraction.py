"""APS extraction prompt templates.

Prompts are stored in ``_PROMPT_DATA`` and exposed via ``__getattr__``
which delegates to the prompt registry for DynamoDB / file resolution.
"""

from __future__ import annotations

# ── Raw prompt data (read by FilePromptBackend) ─────────────────────

_PROMPT_DATA: dict[str, str] = {
    "BATCH_EXTRACTION_PROMPT": """
You are extracting information from an APS (Attending Physician Statement) medical record.
Answer each question based ONLY on the provided context. If the answer is not found, say "Not found".

Context:
{context}

Questions:
{questions}

For each question, return:
{{
    "answers": [
        {{
            "question_id": "<id>",
            "answer": "<extracted answer or 'Not found'>",
            "confidence": <0.0 to 1.0>,
            "citations": [
                {{
                    "page_number": <exact page number from [Page N] markers>,
                    "section_title": "<section name from section headers>",
                    "section_type": "<section type from section headers>",
                    "verbatim_quote": "<exact quote copied from the source text>"
                }}
            ]
        }},
        ...
    ]
}}

IMPORTANT: Each citation must include:
- The exact page number where the evidence appears (use the [Page N] markers in the context)
- The section title and type (from the [Section: ... | Type: ...] headers)
- A verbatim quote copied exactly from the source text — do not paraphrase

Directly return the final JSON structure. Do not output anything else.""",
    "INDIVIDUAL_EXTRACTION_PROMPT": """
You are extracting a specific piece of information from an APS (Attending Physician Statement).
This requires careful analysis and may need cross-referencing across multiple sections.

Context:
{context}

Question: {question}

Think step-by-step:
1. Identify which parts of the context are relevant
2. Cross-reference dates, providers, and findings
3. Synthesize the answer

Return JSON:
{{
    "reasoning": "<step by step analysis>",
    "answer": "<extracted answer or 'Not found'>",
    "confidence": <0.0 to 1.0>,
    "citations": [
        {{
            "page_number": <exact page number from [Page N] markers>,
            "section_title": "<section name from section headers>",
            "section_type": "<section type from section headers>",
            "verbatim_quote": "<exact quote copied from the source text>"
        }}
    ]
}}

IMPORTANT: Each citation must include:
- The exact page number where the evidence appears (use the [Page N] markers in the context)
- The section title and type (from the [Section: ... | Type: ...] headers)
- A verbatim quote copied exactly from the source text — do not paraphrase

Directly return the final JSON structure. Do not output anything else.""",
    "FREE_FORM_CHAT_PROMPT": """Based on the following medical record context, answer the question.

Context:
{context}

Question: {query}

Provide a detailed answer based only on the provided context.""",
}

# ── PEP 562 module __getattr__ ──────────────────────────────────────

_PROMPT_NAMES = frozenset(_PROMPT_DATA.keys())


def __getattr__(name: str) -> str:
    if name in _PROMPT_NAMES:
        from pageindex_rag.prompts.registry import get_prompt

        return get_prompt("aps", "extraction", name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(_PROMPT_NAMES) + ["_PROMPT_DATA"]
