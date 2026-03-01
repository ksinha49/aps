"""Centralized style constants for PDF output formatting."""

from __future__ import annotations

# ── Severity color palette (hex strings) ─────────────────────────────
# Kept as plain hex so the formatter can convert to whatever color object
# the rendering library requires (e.g. reportlab HexColor).

SEVERITY_COLORS: dict[str, str] = {
    "CRITICAL": "#DC2626",
    "SIGNIFICANT": "#D97706",
    "MODERATE": "#CA8A04",
    "MINOR": "#16A34A",
    "INFORMATIONAL": "#2563EB",
}

# ── Risk classification hierarchy ────────────────────────────────────

RISK_CLASSIFICATIONS: list[str] = [
    "Preferred Plus",
    "Preferred",
    "Standard Plus",
    "Standard",
    "Substandard (Table 1-4)",
    "Substandard (Table 5-8)",
    "Substandard (Table 9-12)",
    "Substandard (Table 13-16)",
    "Postpone",
    "Decline",
]

# ── Category display names ───────────────────────────────────────────
# Maps ExtractionCategory enum values to human-readable titles.

CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "demographics": "Patient Demographics",
    "employment": "Employment Information",
    "medical_history": "Medical History",
    "current_medications": "Current Medications",
    "allergies": "Allergies",
    "vital_signs": "Vital Signs",
    "physical_exam": "Physical Examination",
    "lab_results": "Laboratory Results",
    "imaging_results": "Imaging Results",
    "diagnoses": "Diagnoses",
    "procedures": "Procedures",
    "mental_health": "Mental Health",
    "functional_capacity": "Functional Capacity",
    "treatment_plan": "Treatment Plan",
    "prognosis": "Prognosis",
    "physician_opinion": "Physician Opinion",
}

# ── Risk tier colors (APS Schema v1.0.0) ────────────────────────────

RISK_TIER_COLORS: dict[str, str] = {
    "Preferred Plus": "#16A34A",
    "Preferred": "#22C55E",
    "Standard Plus": "#84CC16",
    "Standard": "#2563EB",
    "Substandard": "#D97706",
    "Postpone": "#DC2626",
    "Decline": "#7F1D1D",
}

# ── Lab result flag colors ──────────────────────────────────────────

LAB_FLAG_COLORS: dict[str, str] = {
    "H": "#DC2626",
    "L": "#2563EB",
    "C": "#D97706",
    "": "#16A34A",
}

# ── APS section ordering and titles ─────────────────────────────────

APS_SECTION_ORDER: list[str] = [
    "demographics",
    "build_and_vitals",
    "medical_history",
    "surgical_history",
    "family_history",
    "social_history",
    "mental_health",
    "medications",
    "allergies",
    "lab_results",
    "imaging_and_diagnostics",
    "functional_status",
    "encounter_chronology",
    "red_flags",
    "extraction_summary",
]

APS_SECTION_TITLES: dict[str, str] = {
    "demographics": "Patient Demographics",
    "build_and_vitals": "Build & Vitals",
    "medical_history": "Medical History",
    "surgical_history": "Surgical History",
    "family_history": "Family History",
    "social_history": "Social History",
    "mental_health": "Mental Health",
    "medications": "Medications",
    "allergies": "Allergies",
    "lab_results": "Laboratory Results",
    "imaging_and_diagnostics": "Imaging & Diagnostics",
    "functional_status": "Functional Status",
    "encounter_chronology": "Encounter Chronology",
    "red_flags": "Red Flags & Alerts",
    "extraction_summary": "Extraction Summary",
}

# ── Layout constants ─────────────────────────────────────────────────

HEADER_BG_COLOR = "#1E3A5F"
HEADER_TEXT_COLOR = "#FFFFFF"
SECTION_BORDER_COLOR = "#CBD5E1"
RISK_BOX_BG_COLOR = "#F8FAFC"
ASSESSMENT_BOX_BG_COLOR = "#F0F9FF"

# ── APS-specific layout colors ──────────────────────────────────────

RED_FLAG_BG_COLOR = "#FEF2F2"
RED_FLAG_BORDER_COLOR = "#DC2626"
CITATION_TEXT_COLOR = "#6B7280"
DEMOGRAPHICS_HEADER_BG = "#F1F5F9"
TOC_LINK_COLOR = "#1E3A5F"

# ── Underwriting Y/N condition colors ──────────────────────────────

YN_COLORS: dict[str, str] = {
    "Y": "#DC2626",   # Red
    "N": "#16A34A",   # Green
    "Unknown": "#D97706",  # Amber
}

# ── Underwriting category display names ────────────────────────────

UNDERWRITING_CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    **CATEGORY_DISPLAY_NAMES,
    "encounter_history": "Encounter History",
    "substance_use_history": "Substance Use History",
    "critical_medical_conditions": "Critical Medical Conditions",
    "mental_health_conditions": "Mental Health Conditions",
    "other_critical_conditions": "Other Critical Conditions",
    "morbidity_concerns": "Morbidity Concerns",
    "mortality_concerns": "Mortality Concerns",
    "residence_travel": "Residence & Travel",
}

# ── Underwriting Y/N condition display names ───────────────────────

YN_CONDITION_DISPLAY_NAMES: dict[str, str] = {
    "Alcohol Treatment": "Alcohol Treatment (past 2 yrs)",
    "Tobacco Use": "Tobacco Use",
    "Drug Use/Treatment": "Drug Use/Treatment (past 3 yrs)",
    "Dementia": "Dementia",
    "CVA/Stroke": "CVA/Stroke (past 1 yr)",
    "Myocardial Infarction": "MI/Heart Attack (past 3 mo)",
    "Renal Dialysis": "Renal Dialysis",
    "Cancer": "Cancer (any type)",
    "Cardiac Valve Replacement": "Cardiac Valve Replacement",
    "AIDS/HIV": "AIDS/HIV",
    "Disability (Mental Disorder)": "Disability (Mental Disorder)",
    "Suicide Attempt": "Suicide Attempt (past 1 yr)",
    "Psychiatric Hospitalization": "Psychiatric Hospitalization",
    "Cirrhosis": "Cirrhosis",
    "Gastric Bypass": "Gastric Bypass (past 6 mo)",
    "Foreign Residence/Travel": "Foreign Residence/Travel",
    "Travel-Related Concerns": "Travel-Related Concerns",
}
