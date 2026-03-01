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
    "SYNTHESIS_STRUCTURED_PROMPT": """Given the following extraction results organized by \
category, produce a richly structured APS underwriter summary.

Extraction Results:
{category_summaries}

Document Metadata: {document_metadata}

Return a JSON object with this exact structure:
{{
    "demographics": {{
        "full_name": "<patient full name>",
        "date_of_birth": "<DOB>",
        "age": "<age>",
        "gender": "<gender>",
        "ssn_last4": "<last 4 SSN digits if available>",
        "address": "<address>",
        "phone": "<phone>",
        "insurance_id": "<insurance ID>",
        "employer": "<employer>",
        "occupation": "<occupation>"
    }},
    "sections": [
        {{
            "section_key": "<one of: demographics, build_and_vitals, medical_history, \
surgical_history, family_history, social_history, mental_health, medications, allergies, \
lab_results, imaging_and_diagnostics, functional_status, encounter_chronology, \
red_flags, extraction_summary>",
            "title": "<human-readable section title>",
            "content": "<narrative summary for this section>",
            "source_categories": ["<extraction category values used>"],
            "findings": [
                {{
                    "text": "<clinical finding description>",
                    "severity": "<CRITICAL|SIGNIFICANT|MODERATE|MINOR|INFORMATIONAL>",
                    "citations": [
                        {{
                            "page_number": 0,
                            "date": "<date if known>",
                            "source_type": "<Progress Note|Lab Report|Imaging|etc.>"
                        }}
                    ]
                }}
            ],
            "conditions": [
                {{
                    "name": "<condition name>",
                    "icd10_code": "<ICD-10 code>",
                    "onset_date": "<onset date>",
                    "status": "<active|resolved|chronic>",
                    "severity": "<severity>"
                }}
            ],
            "medications": [
                {{
                    "name": "<drug name>",
                    "dose": "<dosage>",
                    "frequency": "<frequency>",
                    "route": "<oral|IV|etc.>",
                    "prescriber": "<prescriber>",
                    "start_date": "<start date>"
                }}
            ],
            "lab_results": [
                {{
                    "test_name": "<test>",
                    "value": "<result>",
                    "unit": "<unit>",
                    "reference_range": "<range>",
                    "flag": "<H|L|C or empty for normal>",
                    "date": "<date>"
                }}
            ],
            "imaging_results": [
                {{
                    "modality": "<X-ray|MRI|CT|etc.>",
                    "body_part": "<body part>",
                    "finding": "<finding>",
                    "impression": "<impression>",
                    "date": "<date>"
                }}
            ],
            "encounters": [
                {{
                    "date": "<encounter date>",
                    "provider": "<provider name>",
                    "encounter_type": "<office visit|ER|telehealth|etc.>",
                    "summary": "<brief summary>"
                }}
            ],
            "vital_signs": [
                {{
                    "name": "<BP|HR|Temp|etc.>",
                    "value": "<value with unit>",
                    "date": "<date>",
                    "flag": "<H|L or empty>"
                }}
            ],
            "allergies": [
                {{
                    "allergen": "<allergen>",
                    "reaction": "<reaction>",
                    "severity": "<mild|moderate|severe>"
                }}
            ],
            "surgical_history": [
                {{
                    "procedure": "<procedure>",
                    "date": "<date>",
                    "outcome": "<outcome>",
                    "complications": "<complications if any>"
                }}
            ]
        }}
    ],
    "risk_classification": {{
        "tier": "<Preferred Plus|Preferred|Standard Plus|Standard|Substandard|Postpone|Decline>",
        "table_rating": "<Table rating if substandard, e.g. Table 2>",
        "debit_credits": "<debit/credit adjustments>",
        "rationale": "<detailed rationale for the classification>"
    }},
    "risk_factors": ["<identified risk factors>"],
    "red_flags": [
        {{
            "description": "<red flag description>",
            "severity": "<CRITICAL|SIGNIFICANT|MODERATE>",
            "category": "<medication|behavioral|clinical|administrative>"
        }}
    ],
    "overall_assessment": "<comprehensive underwriting assessment paragraph>"
}}

Rules:
- Only include typed data lists (conditions, medications, lab_results, etc.) in sections \
where they are relevant. Omit empty lists.
- Severity must be one of: CRITICAL, SIGNIFICANT, MODERATE, MINOR, INFORMATIONAL.
- Include citations from the extraction results where available.
- Produce sections in order: demographics, build_and_vitals, medical_history, \
surgical_history, family_history, social_history, mental_health, medications, allergies, \
lab_results, imaging_and_diagnostics, functional_status, encounter_chronology.
- Omit sections with no relevant findings.

Directly return the final JSON structure. Do not output anything else.""",
}

# ── PEP 562 module __getattr__ ──────────────────────────────────────

_PROMPT_NAMES = frozenset(_PROMPT_DATA.keys())


def __getattr__(name: str) -> str:
    if name in _PROMPT_NAMES:
        from scout_ai.prompts.registry import get_prompt

        return get_prompt("aps", "synthesis", name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(_PROMPT_NAMES) + ["_PROMPT_DATA"]
