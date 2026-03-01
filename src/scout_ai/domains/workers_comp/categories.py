"""Workers' comp extraction categories with descriptions."""

from __future__ import annotations

from scout_ai.domains.workers_comp.models import ExtractionCategory

CATEGORY_DESCRIPTIONS: dict[ExtractionCategory, str] = {
    ExtractionCategory.CLAIMANT_INFO: (
        "Claimant identification: name, date of birth, social security number, "
        "address, phone number, occupation at time of injury."
    ),
    ExtractionCategory.EMPLOYER_INFO: (
        "Employer name, address, nature of business, supervisor name, "
        "date of hire, employment status, job title and duties."
    ),
    ExtractionCategory.INJURY_DETAILS: (
        "Date, time, and location of injury, mechanism of injury, "
        "body parts affected, witnesses, initial symptoms."
    ),
    ExtractionCategory.MEDICAL_TREATMENT: (
        "Emergency treatment, hospital admissions, surgeries, physical therapy, "
        "medications prescribed, treating physicians, treatment dates."
    ),
    ExtractionCategory.DIAGNOSES: (
        "Primary and secondary diagnoses, ICD-10 codes, "
        "work-related vs. pre-existing conditions."
    ),
    ExtractionCategory.WORK_RESTRICTIONS: (
        "Physical restrictions: lifting, standing, sitting, reaching, "
        "modified duty recommendations, accommodations needed."
    ),
    ExtractionCategory.DISABILITY_STATUS: (
        "Temporary total disability, temporary partial disability, "
        "permanent total/partial disability, disability duration."
    ),
    ExtractionCategory.RETURN_TO_WORK: (
        "Return to work date (actual or estimated), full duty vs modified, "
        "transitional work program, vocational rehabilitation."
    ),
    ExtractionCategory.CAUSATION: (
        "Medical causation opinions, work-relatedness determination, "
        "apportionment between work injury and pre-existing conditions."
    ),
    ExtractionCategory.PRIOR_INJURIES: (
        "Prior workers' comp claims, pre-existing conditions, "
        "prior injuries to same body parts, prior treatments."
    ),
    ExtractionCategory.IMPAIRMENT_RATING: (
        "AMA Guides impairment rating, whole person impairment percentage, "
        "maximum medical improvement date, permanent restrictions."
    ),
    ExtractionCategory.BENEFITS: (
        "Temporary disability benefits, permanent disability benefits, "
        "medical expense reimbursement, vocational rehabilitation benefits."
    ),
}
