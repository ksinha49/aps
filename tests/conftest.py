"""Shared fixtures for pageindex-rag tests."""

from __future__ import annotations

import pytest

from pageindex_rag.config import PageIndexSettings
from pageindex_rag.models import PageContent, TreeNode, MedicalSectionType


@pytest.fixture
def settings() -> PageIndexSettings:
    """Default test settings (approximate tokenizer, no real LLM)."""
    return PageIndexSettings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        tokenizer_method="approximate",
    )


@pytest.fixture
def sample_pages() -> list[PageContent]:
    """10-page synthetic APS document."""
    texts = [
        "FACE SHEET\nPatient Name: John Doe\nDOB: 01/15/1960\nSSN: XXX-XX-1234\nAddress: 123 Main St",
        "HISTORY AND PHYSICAL\nChief Complaint: Lower back pain\nHPI: 55yo male with chronic lower back pain",
        "PROGRESS NOTE - 2024-01-15\nVitals: BP 130/85, HR 72, Temp 98.6F\nAssessment: Chronic lumbar radiculopathy",
        "PROGRESS NOTE - 2024-02-10\nVitals: BP 128/82, HR 70\nPatient reports improvement with physical therapy",
        "LABORATORY REPORT\nCBC: WBC 7.2, RBC 4.8, Hgb 14.2\nCMP: Glucose 95, BUN 18, Creatinine 1.0",
        "IMAGING REPORT\nMRI Lumbar Spine: L4-L5 disc herniation\nModerate foraminal stenosis at L5-S1",
        "MEDICATION LIST\nIbuprofen 800mg TID\nGabapentin 300mg TID\nCyclobenzaprine 10mg QHS",
        "PHYSICAL THERAPY NOTES\nSession 8 of 12\nRange of motion improved 15 degrees extension",
        "OPERATIVE REPORT\nProcedure: L4-L5 microdiscectomy\nAnesthesia: General\nFindings: Large central disc herniation",
        "DISCHARGE SUMMARY\nAdmitted: 03/15/2024\nDischarged: 03/17/2024\nCondition at discharge: Stable, improved",
    ]
    return [PageContent(page_number=i + 1, text=t) for i, t in enumerate(texts)]


@pytest.fixture
def sample_tree() -> list[TreeNode]:
    """Pre-built 3-node tree for utility tests."""
    child1 = TreeNode(
        node_id="0001",
        title="Progress Notes",
        start_index=3,
        end_index=4,
        content_type=MedicalSectionType.PROGRESS_NOTE,
    )
    child2 = TreeNode(
        node_id="0002",
        title="Lab Report",
        start_index=5,
        end_index=5,
        content_type=MedicalSectionType.LAB_REPORT,
    )
    root = TreeNode(
        node_id="0000",
        title="Patient Record",
        start_index=1,
        end_index=10,
        children=[child1, child2],
    )
    return [root]
