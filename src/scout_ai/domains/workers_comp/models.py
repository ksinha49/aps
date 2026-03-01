"""Workers' compensation domain models."""

from __future__ import annotations

from enum import Enum


class SectionType(str, Enum):
    """Recognized workers' comp document section types."""

    FIRST_REPORT = "first_report"
    PROGRESS_NOTE = "progress_note"
    INDEPENDENT_MEDICAL_EXAM = "independent_medical_exam"
    FUNCTIONAL_CAPACITY_EVAL = "functional_capacity_eval"
    EMPLOYER_STATEMENT = "employer_statement"
    WITNESS_STATEMENT = "witness_statement"
    INVESTIGATION_REPORT = "investigation_report"
    BILLING_RECORD = "billing_record"
    CORRESPONDENCE = "correspondence"
    LEGAL_DOCUMENT = "legal_document"
    UNKNOWN = "unknown"


class ExtractionCategory(str, Enum):
    """Workers' comp extraction categories."""

    CLAIMANT_INFO = "claimant_info"
    EMPLOYER_INFO = "employer_info"
    INJURY_DETAILS = "injury_details"
    MEDICAL_TREATMENT = "medical_treatment"
    DIAGNOSES = "diagnoses"
    WORK_RESTRICTIONS = "work_restrictions"
    DISABILITY_STATUS = "disability_status"
    RETURN_TO_WORK = "return_to_work"
    CAUSATION = "causation"
    PRIOR_INJURIES = "prior_injuries"
    IMPAIRMENT_RATING = "impairment_rating"
    BENEFITS = "benefits"
