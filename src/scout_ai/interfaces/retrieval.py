"""Abstract retrieval provider."""

from __future__ import annotations

from abc import ABC, abstractmethod

from scout_ai.models import (
    DocumentIndex,
    ExtractionQuestion,
    RetrievalResult,
)


class IRetrievalProvider(ABC):
    @abstractmethod
    async def retrieve(
        self,
        index: DocumentIndex,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Single-query tree search returning relevant nodes."""

    @abstractmethod
    async def batch_retrieve(
        self,
        index: DocumentIndex,
        questions: list[ExtractionQuestion],
    ) -> dict[str, RetrievalResult]:
        """Category-batched multi-question retrieval."""
