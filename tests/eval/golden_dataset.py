"""Pydantic models for retrieval and extraction evaluation golden datasets."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class GoldenRetrievalCase(BaseModel):
    """A single retrieval evaluation case with expected results."""

    doc_id: str
    query: str
    category: str
    expected_node_ids: list[str] = Field(default_factory=list)
    expected_page_range: tuple[int, int] | None = None


class GoldenExtractionCase(BaseModel):
    """A single extraction evaluation case with expected answer and tolerance."""

    doc_id: str
    question_id: str
    category: str
    expected_answer: str
    tolerance: str = "exact"  # exact | contains | numeric_within_5pct


class GoldenDataset(BaseModel):
    """Collection of golden retrieval and extraction cases for a domain."""

    domain: str
    version: str = "1.0"
    retrieval_cases: list[GoldenRetrievalCase] = Field(default_factory=list)
    extraction_cases: list[GoldenExtractionCase] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> GoldenDataset:
        """Load a golden dataset from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
