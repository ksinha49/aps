"""APS extraction categories with descriptions for prompt context."""

from __future__ import annotations

from scout_ai.domains.aps.models import ExtractionCategory

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

# ── Underwriting-specific category descriptions ────────────────────
# Tailored for the 50-question underwriting template.  Keyed by plain
# string because several categories (e.g. substance_use_history) are
# not in the base ExtractionCategory enum.

UNDERWRITING_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    # Overrides for existing categories (underwriting-focused wording)
    "demographics": (
        "Patient identification: full name, date of birth, gender, "
        "policy number, insurance ID, date range of the APS records."
    ),
    "vital_signs": (
        "All blood pressure readings with dates, BMI with date, height and weight "
        "with dates, blood pressure trend analysis (improving, worsening, stable)."
    ),
    "lab_results": (
        "All laboratory test results with values, reference ranges, and normal/abnormal "
        "flags. Critical or significantly abnormal values. HbA1c, lipid panel "
        "(total cholesterol, LDL, HDL, triglycerides) with dates."
    ),
    "current_medications": (
        "All medications with dosage, frequency, and prescribing physician. "
        "Whether each medication is currently being taken. "
        "Controlled substances (opioids, benzodiazepines, stimulants)."
    ),
    "diagnoses": (
        "All diagnosed conditions with dates of onset, chronic vs. acute status, "
        "current status. Specialist referrals with specialist type, reason, and date."
    ),
    "procedures": (
        "All surgical and non-surgical procedures with dates and outcomes. "
        "Complications from any procedures."
    ),
    "medical_history": (
        "Complete past medical and surgical history summary. "
        "Additional relevant history including family history, social history, "
        "and review of systems."
    ),
    # New categories
    "encounter_history": (
        "All clinical encounters with visit dates and reasons. "
        "Most recent encounter date and reason. Total number of encounters "
        "documented in the APS."
    ),
    "substance_use_history": (
        "Alcohol use, tobacco use, illicit drug use. Treatment for alcohol "
        "abuse within 2 years. Current or historical tobacco use. Drug use "
        "or treatment for drug abuse within 3 years. Look in Social History "
        "sections, intake forms, and progress notes."
    ),
    "critical_medical_conditions": (
        "Dementia or cognitive impairment. CVA/stroke or TIA within 1 year. "
        "Myocardial infarction within 3 months. Renal dialysis. Cancer (any type) "
        "with type, stage, and treatment status. Cardiac valve replacement or repair. "
        "AIDS/HIV diagnosis. Dates, treatments, and current status for each."
    ),
    "mental_health_conditions": (
        "Disability due to mental health disorder. Suicide attempts within 1 year. "
        "Psychiatric hospitalizations with dates and reasons. "
        "Look in Mental Health, Psychiatry, and Behavioral Health sections."
    ),
    "other_critical_conditions": (
        "Cirrhosis of the liver. Gastric bypass or bariatric surgery within "
        "6 months. Look in Diagnoses, Surgical History, and GI sections."
    ),
    "morbidity_concerns": (
        "Morbidity risk factors: uncontrolled conditions, multiple comorbidities, "
        "medication non-compliance, progressive disease. Future morbidity risk "
        "assessment based on current medical findings."
    ),
    "mortality_concerns": (
        "Mortality risk factors: life-threatening conditions, terminal diagnoses, "
        "advanced disease staging, organ failure, end-of-life indicators."
    ),
    "residence_travel": (
        "Foreign residence or travel history. Travel-related health concerns "
        "including tropical diseases, endemic exposures. Look in Social History, "
        "Travel History, and intake forms."
    ),
}
