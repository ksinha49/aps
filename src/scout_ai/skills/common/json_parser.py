"""JSON parsing skill — extracts structured JSON from LLM responses."""

from __future__ import annotations

import json
import logging
from typing import Any

from strands import tool

log = logging.getLogger(__name__)


@tool
def parse_json(content: str) -> str:
    """Parse JSON from an LLM response, handling common issues like fences and trailing commas.

    Args:
        content: Raw LLM response text that may contain JSON wrapped in code fences.

    Returns:
        The parsed JSON as a string, or '{}' if parsing fails.
    """
    result = extract_json(content)
    return json.dumps(result)


def extract_json(content: str) -> Any:
    """Parse JSON from LLM response, handling fences and common issues.

    This is the pure-logic function — usable without the Strands @tool wrapper.
    """
    try:
        start = content.find("```json")
        if start != -1:
            start += 7
            end = content.rfind("```")
            json_str = content[start:end].strip()
        else:
            json_str = content.strip()

        json_str = json_str.replace("None", "null")
        json_str = json_str.replace("\n", " ").replace("\r", " ")
        json_str = " ".join(json_str.split())

        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            json_str = json_str.replace(",]", "]").replace(",}", "}")  # type: ignore[possibly-undefined]
            return json.loads(json_str)
        except Exception:
            log.error("Failed to parse JSON from LLM response")
            return {}
    except Exception:
        log.error("Unexpected error extracting JSON")
        return {}
