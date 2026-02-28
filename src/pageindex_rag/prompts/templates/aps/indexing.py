"""APS indexing prompt templates.

Consolidates prompts from ``aps/prompts.py`` plus inline prompts from
``indexer.py`` into a single module for the indexing pipeline.

Prompts are stored in ``_PROMPT_DATA`` and exposed via ``__getattr__``
which delegates to the prompt registry for DynamoDB / file resolution.
"""

from __future__ import annotations

# ── Raw prompt data (read by FilePromptBackend) ─────────────────────

_PROMPT_DATA: dict[str, str] = {
    "TOC_DETECT_PROMPT": """
You will analyze a page from an Attending Physician Statement (APS) or medical record.
Determine if this page contains a table of contents or an index listing sections with page references.

Given text: {content}

Return JSON:
{{
    "thinking": "<your reasoning>",
    "toc_detected": "<yes or no>"
}}

Note: Medication lists, lab results, and problem lists are NOT tables of contents.
Directly return the final JSON structure. Do not output anything else.""",
    "DETECT_PAGE_NUMBERS_PROMPT": """Detect if there are page numbers/indices in this table of contents.

Given text: {toc_content}

Return JSON:
{{"thinking": "<reasoning>", "page_index_given_in_toc": "<yes or no>"}}
Directly return JSON only.""",
    "TOC_TRANSFORM_PROMPT": """Transform this table of contents into JSON format.

structure is the hierarchy index (1, 1.1, 1.2, 2, etc.).

Return JSON:
{{"table_of_contents": [
    {{"structure": "<x.x.x>", "title": "<section title>", "page": <page_number or null>}},
    ...
]}}

Given table of contents:
{toc_content}

Directly return the final JSON structure. Do not output anything else.""",
    "EXTRACT_TOC_INDICES_PROMPT": """Add physical_index to each TOC entry based on where sections appear in the doc.

Pages use <physical_index_X> tags. Only add physical_index for sections found in provided pages.

Table of contents: {toc_no_page}
Document pages: {content}

Return JSON array:
[{{"structure": "<>", "title": "<>", "physical_index": "<physical_index_X>"}}]

Directly return JSON only.""",
    "ADD_PAGE_NUMBERS_PROMPT": """Check if sections from the structure appear in the document pages.

Pages use <physical_index_X> tags. Add physical_index for sections found.

Current Document Pages:
{part}

Given Structure:
{structure}

Return the full structure with physical_index added where found.
Directly return JSON only.""",
    "GENERATE_TOC_INIT_PROMPT": """
You are an expert in analyzing medical document structure.
Extract the hierarchical section structure from this portion of an Attending Physician Statement (APS).

Medical documents typically contain sections like: Face Sheet, History & Physical,
Progress Notes, Lab Reports, Imaging Reports, Operative Reports, Discharge Summaries,
Medication Lists, Consultation Notes, Physical Therapy Notes, etc.

The structure code uses dot notation for hierarchy (1, 1.1, 1.2, 2, 2.1, etc.).
The text contains tags like <physical_index_X> marking page boundaries.

Extract the physical_index of where each section starts.

Return JSON array:
[
    {{
        "structure": "<hierarchy code>",
        "title": "<section title from the text>",
        "physical_index": "<physical_index_X>"
    }},
    ...
]

Given text:
{part}

Directly return the final JSON structure. Do not output anything else.""",
    "GENERATE_TOC_CONTINUE_PROMPT": """
You are an expert in analyzing medical document structure.
Continue the hierarchical section structure from the previous part to cover this new section.

Previous tree structure:
{previous_toc}

New text:
{part}

Return JSON array of ONLY the new sections found in the new text:
[
    {{
        "structure": "<hierarchy code continuing from previous>",
        "title": "<section title>",
        "physical_index": "<physical_index_X>"
    }},
    ...
]

Directly return the final JSON structure. Do not output anything else.""",
    "CHECK_TITLE_APPEARANCE_PROMPT": """
Your job is to check if the given section appears or starts in the given page_text.
This is an APS (Attending Physician Statement) / medical record document.

Note: do fuzzy matching, ignore any space inconsistency in the page_text.

The given section title is {title}.
The given page_text is {page_text}.

Reply format:
{{
    "thinking": "<your reasoning>",
    "answer": "yes or no"
}}
Directly return the final JSON structure. Do not output anything else.""",
    "CHECK_TITLE_START_PROMPT": """
Check if the section title appears at the BEGINNING of the page.
If there is other content before the section title, answer "no".

The given section title is {title}.
The given page_text is {page_text}.

Reply format:
{{
    "thinking": "<your reasoning>",
    "start_begin": "yes or no"
}}
Directly return the final JSON structure. Do not output anything else.""",
    "FIX_INCORRECT_TOC_PROMPT": """Find the physical page where this section starts.

Section Title: {title}
Document pages: {search_content}

Return JSON: {{"physical_index": "<physical_index_X>"}}
Directly return JSON only.""",
    "GENERATE_SUMMARY_PROMPT": """
Summarize this section of a medical record (APS document).
Focus on clinically relevant information: diagnoses, findings, measurements, dates.

Section text: {text}

Return a concise 2-3 sentence summary. Directly return the summary text only.""",
    "GENERATE_DOC_DESCRIPTION_PROMPT": """Generate a one-sentence description for this medical document (APS).

Document structure:
{structure_summary}

Return the description only.""",
    "TOC_CONTINUE_PROMPT": """Continue the table of contents JSON structure.

Raw TOC: {toc_content}

Incomplete JSON so far: {content}

Continue directly from where it was cut off.""",
}

# ── PEP 562 module __getattr__ ──────────────────────────────────────

_PROMPT_NAMES = frozenset(_PROMPT_DATA.keys())


def __getattr__(name: str) -> str:
    if name in _PROMPT_NAMES:
        from pageindex_rag.prompts.registry import get_prompt

        return get_prompt("aps", "indexing", name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(_PROMPT_NAMES) + ["_PROMPT_DATA"]
