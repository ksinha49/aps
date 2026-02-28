"""Unit tests for MedicalSectionClassifier (heuristic only, no LLM)."""

import pytest

from pageindex_rag.models import MedicalSectionType, PageContent
from pageindex_rag.providers.pageindex.medical_classifier import MedicalSectionClassifier


@pytest.fixture
def classifier():
    return MedicalSectionClassifier(client=None)


class TestClassifyByTitle:
    @pytest.mark.parametrize(
        "title,expected",
        [
            ("Face Sheet", MedicalSectionType.FACE_SHEET),
            ("FACE SHEET", MedicalSectionType.FACE_SHEET),
            ("Patient Demographics", MedicalSectionType.FACE_SHEET),
            ("Progress Note", MedicalSectionType.PROGRESS_NOTE),
            ("PROGRESS NOTES", MedicalSectionType.PROGRESS_NOTE),
            ("Clinical Note", MedicalSectionType.PROGRESS_NOTE),
            ("Office Visit", MedicalSectionType.PROGRESS_NOTE),
            ("SOAP Note", MedicalSectionType.PROGRESS_NOTE),
            ("Laboratory Report", MedicalSectionType.LAB_REPORT),
            ("Lab Results", MedicalSectionType.LAB_REPORT),
            ("CBC", MedicalSectionType.LAB_REPORT),
            ("Blood Test Results", MedicalSectionType.LAB_REPORT),
            ("Imaging Report", MedicalSectionType.IMAGING),
            ("MRI Lumbar Spine", MedicalSectionType.IMAGING),
            ("CT Scan Report", MedicalSectionType.IMAGING),
            ("Pathology Report", MedicalSectionType.PATHOLOGY),
            ("Biopsy Report", MedicalSectionType.PATHOLOGY),
            ("Operative Report", MedicalSectionType.OPERATIVE_REPORT),
            ("Surgical Note", MedicalSectionType.OPERATIVE_REPORT),
            ("Discharge Summary", MedicalSectionType.DISCHARGE_SUMMARY),
            ("Discharge Note", MedicalSectionType.DISCHARGE_SUMMARY),
            ("Consultation Report", MedicalSectionType.CONSULTATION),
            ("Medication List", MedicalSectionType.MEDICATION_LIST),
            ("Current Medications", MedicalSectionType.MEDICATION_LIST),
            ("Vital Signs", MedicalSectionType.VITAL_SIGNS),
            ("Nursing Note", MedicalSectionType.NURSING_NOTE),
            ("Physical Therapy Notes", MedicalSectionType.THERAPY_NOTE),
            ("Occupational Therapy", MedicalSectionType.THERAPY_NOTE),
            ("Psychiatric Evaluation", MedicalSectionType.MENTAL_HEALTH),
            ("Mental Health Assessment", MedicalSectionType.MENTAL_HEALTH),
            ("Dental Exam Report", MedicalSectionType.DENTAL),
        ],
    )
    def test_known_titles(self, classifier, title, expected):
        assert classifier.classify_by_title(title) == expected

    def test_unknown_title(self, classifier):
        assert classifier.classify_by_title("Random Section") == MedicalSectionType.UNKNOWN

    def test_empty_title(self, classifier):
        assert classifier.classify_by_title("") == MedicalSectionType.UNKNOWN


class TestDetectSectionsHeuristic:
    def test_finds_sections(self, classifier, sample_pages):
        sections = classifier.detect_sections_heuristic(sample_pages)
        # Should find at least some sections from the sample pages
        assert len(sections) >= 2

        titles = [s["title"] for s in sections]
        types = [s["section_type"] for s in sections]

        # Should detect these prominent headers
        assert any("FACE SHEET" in t for t in titles)
        assert MedicalSectionType.FACE_SHEET in types

    def test_empty_pages(self, classifier):
        sections = classifier.detect_sections_heuristic([])
        assert sections == []

    def test_no_medical_content(self, classifier):
        pages = [PageContent(page_number=1, text="this is just random text with no headers")]
        sections = classifier.detect_sections_heuristic(pages)
        assert sections == []

    def test_returns_page_numbers(self, classifier, sample_pages):
        sections = classifier.detect_sections_heuristic(sample_pages)
        for section in sections:
            assert "page_number" in section
            assert isinstance(section["page_number"], int)
            assert 1 <= section["page_number"] <= len(sample_pages)
