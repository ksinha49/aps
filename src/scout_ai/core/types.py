"""Shared type aliases and enums for the framework layer."""

from __future__ import annotations

from typing import Any

# JSON-like dict returned by LLM parsing
JsonDict = dict[str, Any]
JsonList = list[dict[str, Any]]

# Page number → text mapping used for citation context
PageMap = dict[int, str]
