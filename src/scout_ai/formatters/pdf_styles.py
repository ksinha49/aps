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

# ── Layout constants ─────────────────────────────────────────────────

HEADER_BG_COLOR = "#1E3A5F"
HEADER_TEXT_COLOR = "#FFFFFF"
SECTION_BORDER_COLOR = "#CBD5E1"
RISK_BOX_BG_COLOR = "#F8FAFC"
ASSESSMENT_BOX_BG_COLOR = "#F0F9FF"
