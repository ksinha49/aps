"""Abstract chat / extraction provider."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pageindex_rag.models import ExtractionQuestion, ExtractionResult


class IChatProvider(ABC):
    @abstractmethod
    async def extract_answers(
        self,
        questions: list[ExtractionQuestion],
        context: str,
    ) -> list[ExtractionResult]:
        """Extract answers to questions from provided context."""

    @abstractmethod
    async def chat(self, query: str, context: str) -> str:
        """Free-form completion over context."""
