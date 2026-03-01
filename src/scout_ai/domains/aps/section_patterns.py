"""Regex patterns for medical document section detection.

These patterns are used by ``MedicalSectionClassifier`` to identify
APS section types without LLM calls (heuristic-first approach).
"""

from __future__ import annotations

import re
from typing import Pattern

from scout_ai.models import MedicalSectionType

# Each pattern matches common section titles / headers in APS documents.
# Patterns are case-insensitive and match at word boundaries.

SECTION_PATTERNS: dict[MedicalSectionType, list[Pattern[str]]] = {
    MedicalSectionType.FACE_SHEET: [
        re.compile(r"\bface\s*sheet\b", re.IGNORECASE),
        re.compile(r"\bpatient\s+demographics?\b", re.IGNORECASE),
        re.compile(r"\badmission\s+data\b", re.IGNORECASE),
        re.compile(r"\bregistration\s+form\b", re.IGNORECASE),
    ],
    MedicalSectionType.PROGRESS_NOTE: [
        re.compile(r"\bprogress\s+note", re.IGNORECASE),
        re.compile(r"\bclinic(?:al)?\s+note", re.IGNORECASE),
        re.compile(r"\boffice\s+visit\b", re.IGNORECASE),
        re.compile(r"\bsoap\s+note\b", re.IGNORECASE),
        re.compile(r"\bfollow[\s-]?up\s+note\b", re.IGNORECASE),
    ],
    MedicalSectionType.LAB_REPORT: [
        re.compile(r"\blab(?:oratory)?\s+(?:report|result)", re.IGNORECASE),
        re.compile(r"\bblood\s+(?:test|work|panel)", re.IGNORECASE),
        re.compile(r"\b(?:CBC|CMP|BMP|UA)\b"),
        re.compile(r"\bcomplete\s+blood\s+count\b", re.IGNORECASE),
        re.compile(r"\bchemistry\s+panel\b", re.IGNORECASE),
    ],
    MedicalSectionType.IMAGING: [
        re.compile(r"\b(?:imaging|radiology)\s+report\b", re.IGNORECASE),
        re.compile(r"\b(?:MRI|CT|X-?ray|ultrasound)\s", re.IGNORECASE),
        re.compile(r"\bradiolog(?:y|ist)\b", re.IGNORECASE),
    ],
    MedicalSectionType.PATHOLOGY: [
        re.compile(r"\bpathology\s+report\b", re.IGNORECASE),
        re.compile(r"\bbiopsy\s+report\b", re.IGNORECASE),
        re.compile(r"\bhistopatholog", re.IGNORECASE),
        re.compile(r"\bcytolog(?:y|ical)\b", re.IGNORECASE),
    ],
    MedicalSectionType.OPERATIVE_REPORT: [
        re.compile(r"\boperative\s+(?:report|note)\b", re.IGNORECASE),
        re.compile(r"\bsurgical\s+(?:report|note)\b", re.IGNORECASE),
        re.compile(r"\bprocedure\s+(?:report|note)\b", re.IGNORECASE),
        re.compile(r"\bop[\s-]?note\b", re.IGNORECASE),
    ],
    MedicalSectionType.DISCHARGE_SUMMARY: [
        re.compile(r"\bdischarge\s+summar", re.IGNORECASE),
        re.compile(r"\bdischarge\s+note\b", re.IGNORECASE),
        re.compile(r"\bdischarge\s+instruction", re.IGNORECASE),
    ],
    MedicalSectionType.CONSULTATION: [
        re.compile(r"\bconsult(?:ation)?\s+(?:report|note)\b", re.IGNORECASE),
        re.compile(r"\breferral\s+(?:report|note)\b", re.IGNORECASE),
        re.compile(r"\bspecialist\s+consult", re.IGNORECASE),
    ],
    MedicalSectionType.MEDICATION_LIST: [
        re.compile(r"\bmedication\s+(?:list|record|reconciliation)\b", re.IGNORECASE),
        re.compile(r"\bprescription\s+(?:list|record|history)\b", re.IGNORECASE),
        re.compile(r"\bmed(?:ication)?\s+reconcil", re.IGNORECASE),
        re.compile(r"\bcurrent\s+medication", re.IGNORECASE),
    ],
    MedicalSectionType.VITAL_SIGNS: [
        re.compile(r"\bvital\s+signs?\b", re.IGNORECASE),
        re.compile(r"\bvitals?\b", re.IGNORECASE),
        re.compile(r"\bTPR\b"),
    ],
    MedicalSectionType.NURSING_NOTE: [
        re.compile(r"\bnursing\s+(?:note|assessment|record)\b", re.IGNORECASE),
        re.compile(r"\bnurse[''s]*\s+note\b", re.IGNORECASE),
    ],
    MedicalSectionType.THERAPY_NOTE: [
        re.compile(r"\b(?:physical|occupational|speech)\s+therap", re.IGNORECASE),
        re.compile(r"\brehabilitation\s+note\b", re.IGNORECASE),
        re.compile(r"\bPT\s+note\b", re.IGNORECASE),
        re.compile(r"\bOT\s+note\b", re.IGNORECASE),
    ],
    MedicalSectionType.MENTAL_HEALTH: [
        re.compile(r"\bpsychiat", re.IGNORECASE),
        re.compile(r"\bpsycholog", re.IGNORECASE),
        re.compile(r"\bmental\s+health\b", re.IGNORECASE),
        re.compile(r"\bbehavioral\s+health\b", re.IGNORECASE),
    ],
    MedicalSectionType.DENTAL: [
        re.compile(r"\bdental\s+(?:exam|record|note|report)\b", re.IGNORECASE),
        re.compile(r"\bodontolog", re.IGNORECASE),
    ],
}
