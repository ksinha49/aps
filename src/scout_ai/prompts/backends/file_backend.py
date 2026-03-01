"""File-based prompt backend — wraps the existing importlib loading logic.

Each template module stores its prompts in a ``_PROMPT_DATA`` dict. This backend
reads from that dict directly, bypassing module ``__getattr__`` and avoiding
infinite recursion (since ``__getattr__`` delegates to ``get_prompt()`` which
may call back into this backend).
"""

from __future__ import annotations

import importlib
from typing import Any

from scout_ai.prompts.context import PromptContext


class FilePromptBackend:
    """Loads prompts from Python modules on disk via importlib.

    Module path convention: ``scout_ai.prompts.templates.{domain}.{category}``

    Each module must expose a ``_PROMPT_DATA: dict[str, str]`` mapping
    constant names to their template strings.
    """

    def __init__(self) -> None:
        self._modules: dict[tuple[str, str], Any] = {}

    def get(
        self,
        domain: str,
        category: str,
        name: str,
        *,
        context: PromptContext | None = None,
        version: int | None = None,
    ) -> str:
        """Load a prompt from ``_PROMPT_DATA`` in ``prompts/templates/{domain}/{category}.py``.

        The ``context`` and ``version`` parameters are accepted for interface
        compatibility but are ignored — the file backend has no dimension or
        version awareness.
        """
        key = (domain, category)
        if key not in self._modules:
            module_path = f"scout_ai.prompts.templates.{domain}.{category}"
            try:
                self._modules[key] = importlib.import_module(module_path)
            except ModuleNotFoundError as exc:
                raise KeyError(f"Prompt module not found: {module_path}") from exc

        module = self._modules[key]
        data: dict[str, str] | None = getattr(module, "_PROMPT_DATA", None)
        if data is not None and name in data:
            return data[name]

        raise KeyError(f"Prompt {name!r} not found in {domain}/{category}")
