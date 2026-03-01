"""Medical section detection: heuristic-first with LLM fallback.

Regex patterns detect 80-90% of APS section types without LLM calls.
LLM fallback only fires for ambiguous titles/content.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from scout_ai.domains.aps.section_patterns import SECTION_PATTERNS
from scout_ai.models import MedicalSectionType, PageContent
from scout_ai.prompts.templates.aps.classification import CLASSIFY_SECTION_PROMPT

log = logging.getLogger(__name__)


class MedicalSectionClassifier:
    """Classifies APS document sections by type using regex-first strategy."""

    def __init__(self, client: Optional[Any] = None) -> None:
        self._client = client

    def classify_by_title(self, title: str) -> MedicalSectionType:
        """Fast regex classification from section title alone."""
        for section_type, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(title):
                    return section_type
        return MedicalSectionType.UNKNOWN

    async def classify_by_content(
        self, title: str, text: str
    ) -> MedicalSectionType:
        """LLM-based classification for ambiguous sections."""
        if self._client is None:
            return MedicalSectionType.UNKNOWN

        prompt = CLASSIFY_SECTION_PROMPT.format(
            title=title, content_preview=text[:500]
        )
        response = await self._client.complete(prompt)
        parsed = self._client.extract_json(response)
        section_type_str = parsed.get("section_type", "unknown")

        try:
            return MedicalSectionType(section_type_str)
        except ValueError:
            return MedicalSectionType.UNKNOWN

    async def classify(self, title: str, text: str = "") -> MedicalSectionType:
        """Classify with heuristic-first, LLM fallback."""
        result = self.classify_by_title(title)
        if result != MedicalSectionType.UNKNOWN:
            return result
        if text:
            return await self.classify_by_content(title, text)
        return MedicalSectionType.UNKNOWN

    def detect_sections_heuristic(
        self, pages: list[PageContent]
    ) -> list[dict[str, Any]]:
        """Detect section boundaries using regex patterns on page content.

        Returns a list of detected section boundaries with:
        - ``title``: matched section header text
        - ``section_type``: the ``MedicalSectionType``
        - ``page_number``: 1-indexed page where the section starts
        """
        header_pattern = re.compile(
            r"^[\s]*([A-Z][A-Z\s\-&/]{3,}(?:REPORT|NOTE|SUMMARY|LIST|SHEET|SIGNS|RECORD)?)\s*$",
            re.MULTILINE,
        )
        sections: list[dict[str, Any]] = []
        seen_titles: set[str] = set()

        for page in pages:
            matches = header_pattern.finditer(page.text)
            for match in matches:
                raw_title = match.group(1).strip()
                section_type = self.classify_by_title(raw_title)
                if section_type != MedicalSectionType.UNKNOWN:
                    key = f"{section_type.value}:{raw_title}"
                    if key not in seen_titles:
                        seen_titles.add(key)
                        sections.append({
                            "title": raw_title,
                            "section_type": section_type,
                            "page_number": page.page_number,
                        })

        log.info(f"Heuristic section detection found {len(sections)} sections")
        return sections
