"""APS synthesis prompt templates for underwriter summary generation.

Prompts are stored in ``_PROMPT_DATA`` and exposed via ``__getattr__``
which delegates to the prompt registry for DynamoDB / file resolution.
"""

from __future__ import annotations

# ── Raw prompt data (read by FilePromptBackend) ─────────────────────

_PROMPT_DATA: dict[str, str] = {
    "SYNTHESIS_SYSTEM_PROMPT": """You are an expert medical underwriter synthesizing \
extraction results from an Attending Physician Statement (APS) into a structured \
summary report.

Your task is to produce a clear, professional underwriter summary that:
1. Highlights clinically significant findings organized by medical domain
2. Identifies risk factors relevant to life/health insurance underwriting
3. Flags contradictions or gaps in the medical record
4. Provides an overall risk assessment based on the available evidence
5. Uses only information present in the extraction results — do not infer or fabricate

Always cite the extraction category and question ID when referencing specific findings.""",
    "SYNTHESIS_PROMPT": """Given the following extraction results organized by category, \
produce a structured underwriter summary report.

Extraction Results:
{category_summaries}

Document Metadata: {document_metadata}

Return a JSON object with this exact structure:
{{
    "patient_demographics": "<summary of patient identifying information>",
    "sections": [
        {{
            "title": "<section title, e.g. 'Medical History', 'Current Conditions'>",
            "content": "<narrative summary of findings in this domain>",
            "source_categories": ["<category values used>"],
            "key_findings": ["<bullet point findings>"]
        }}
    ],
    "risk_factors": ["<identified risk factors for underwriting>"],
    "overall_assessment": "<professional underwriting assessment paragraph>"
}}

Produce sections for: Patient Demographics, Medical History, Current Conditions & Diagnoses, \
Medications & Treatment, Laboratory & Imaging Results, Functional Status, \
Mental Health, and Prognosis & Physician Opinion.

Directly return the final JSON structure. Do not output anything else.""",
}

# ── PEP 562 module __getattr__ ──────────────────────────────────────

_PROMPT_NAMES = frozenset(_PROMPT_DATA.keys())


def __getattr__(name: str) -> str:
    if name in _PROMPT_NAMES:
        from pageindex_rag.prompts.registry import get_prompt

        return get_prompt("aps", "synthesis", name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(_PROMPT_NAMES) + ["_PROMPT_DATA"]
