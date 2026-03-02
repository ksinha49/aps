"""Pluggable token counter with three backends.

Modes:
  - ``approximate``: chars / 4 (no dependencies, fast)
  - ``tiktoken``: OpenAI tiktoken (requires ``tiktoken`` extra)
  - ``transformers``: HuggingFace tokenizers (requires ``transformers`` extra)
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from scout_ai.exceptions import TokenizerError

log = logging.getLogger(__name__)

# Cache for tiktoken encoders
_tiktoken_cache: dict[str, object] = {}


class TokenCounter:
    """Count tokens using the configured method."""

    def __init__(
        self,
        method: Literal["approximate", "tiktoken", "transformers"] = "approximate",
        model: str = "gpt-4o",
        char_to_token_ratio: int = 4,
        fallback_encoding: str = "cl100k_base",
    ) -> None:
        self.method = method
        self.model = model
        self._char_to_token_ratio = char_to_token_ratio
        self._fallback_encoding = fallback_encoding

        if method == "tiktoken":
            try:
                import tiktoken  # noqa: F401
            except ImportError as e:
                raise TokenizerError(
                    "tiktoken not installed. Install with: pip install scout-ai[tiktoken]"
                ) from e
        elif method == "transformers":
            try:
                import transformers  # noqa: F401
            except ImportError as e:
                raise TokenizerError(
                    "transformers not installed. Install with: pip install scout-ai[transformers]"
                ) from e

    def count(self, text: str, model: Optional[str] = None) -> int:
        """Return the token count for *text*."""
        if not text:
            return 0

        effective_model = model or self.model

        if self.method == "approximate":
            return self._count_approximate(text)
        elif self.method == "tiktoken":
            return self._count_tiktoken(text, effective_model)
        else:
            return self._count_transformers(text, effective_model)

    # ── Backends ─────────────────────────────────────────────────────

    def _count_approximate(self, text: str) -> int:
        return max(1, len(text) // self._char_to_token_ratio)

    def _count_tiktoken(self, text: str, model: str) -> int:
        import tiktoken

        cache_key = f"{model}:{self._fallback_encoding}"
        if cache_key not in _tiktoken_cache:
            try:
                _tiktoken_cache[cache_key] = tiktoken.encoding_for_model(model)
            except KeyError:
                _tiktoken_cache[cache_key] = tiktoken.get_encoding(self._fallback_encoding)
        enc = _tiktoken_cache[cache_key]
        return len(enc.encode(text))  # type: ignore[union-attr]

    @staticmethod
    def _count_transformers(text: str, model: str) -> int:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model)
        return len(tokenizer.encode(text))
