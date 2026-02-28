"""APS extraction categories with descriptions for prompt context."""

from __future__ import annotations

from pageindex_rag.models import ExtractionCategory

CATEGORY_DESCRIPTIONS: dict[ExtractionCategory, str] = {
    ExtractionCategory.DEMOGRAPHICS: (
        "Patient identification: name, date of birth, social security number, "
        "address, phone number, emergency contacts, insurance information."
    ),
    ExtractionCategory.EMPLOYMENT: (
        "Employment status, occupation, employer name, work schedule, "
        "job duties, date last worked, disability status."
    ),
    ExtractionCategory.MEDICAL_HISTORY: (
        "Past medical/surgical history, family history, social history (tobacco, "
        "alcohol, drug use), review of systems, prior hospitalizations."
    ),
    ExtractionCategory.CURRENT_MEDICATIONS: (
        "Active prescriptions, over-the-counter medications, supplements, "
        "dosages, frequencies, routes of administration, prescribing physicians."
    ),
    ExtractionCategory.ALLERGIES: (
        "Drug allergies, food allergies, environmental allergies, "
        "adverse drug reactions, allergy severity and reactions."
    ),
    ExtractionCategory.VITAL_SIGNS: (
        "Blood pressure, heart rate, respiratory rate, temperature, "
        "oxygen saturation, weight, height, BMI, pain scale."
    ),
    ExtractionCategory.PHYSICAL_EXAM: (
        "Physical examination findings by system: general appearance, HEENT, "
        "cardiovascular, respiratory, abdominal, musculoskeletal, neurological, skin."
    ),
    ExtractionCategory.LAB_RESULTS: (
        "Laboratory test results: CBC, CMP, lipid panel, thyroid function, "
        "urinalysis, HbA1c, PSA, liver function tests, coagulation studies."
    ),
    ExtractionCategory.IMAGING_RESULTS: (
        "Diagnostic imaging findings: X-ray, MRI, CT scan, ultrasound, "
        "PET scan, bone density scan, echocardiogram, angiography."
    ),
    ExtractionCategory.DIAGNOSES: (
        "Primary and secondary diagnoses, ICD-10 codes, differential diagnoses, "
        "chief complaints, date of onset, chronicity."
    ),
    ExtractionCategory.PROCEDURES: (
        "Surgical procedures, non-surgical procedures, biopsies, injections, "
        "dates performed, findings, complications, outcomes."
    ),
    ExtractionCategory.MENTAL_HEALTH: (
        "Psychiatric diagnoses, mental status exam, PHQ-9/GAD-7 scores, "
        "behavioral observations, cognitive assessments, suicide risk."
    ),
    ExtractionCategory.FUNCTIONAL_CAPACITY: (
        "Activities of daily living (ADLs), instrumental ADLs, mobility, "
        "functional capacity evaluation, work restrictions, disability rating."
    ),
    ExtractionCategory.TREATMENT_PLAN: (
        "Recommended treatments, referrals, follow-up schedule, "
        "physical therapy plan, medication changes, lifestyle modifications."
    ),
    ExtractionCategory.PROGNOSIS: (
        "Expected recovery timeline, disease progression, return-to-work estimate, "
        "permanent impairment rating, maximum medical improvement date."
    ),
    ExtractionCategory.PHYSICIAN_OPINION: (
        "Attending physician's opinion on causation, work-relatedness, "
        "ability to perform job duties, need for ongoing treatment, "
        "degree of impairment, future medical needs."
    ),
}
