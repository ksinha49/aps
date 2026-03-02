"""LLMLingua compressor — token-level compression using LLMLingua-2.

Requires optional dependency: ``pip install scout-ai[llmlingua]``
"""

from __future__ import annotations

from scout_ai.context.models import CompressedContext


class LLMLinguaCompressor:
    """Token-level compression using LLMLingua-2.

    Achieves 2-5x compression ratios with minimal information loss
    by scoring individual tokens for their contribution to meaning.

    Requires: ``pip install scout-ai[llmlingua]``
    """

    def __init__(self, target_ratio: float = 0.5, device: str = "cpu") -> None:
        self._target_ratio = target_ratio
        self._device = device
        self._compressor: object | None = None

    def _get_compressor(self) -> object:
        """Lazy-initialize the LLMLingua PromptCompressor."""
        if self._compressor is not None:
            return self._compressor
        try:
            from llmlingua import PromptCompressor  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "LLMLingua is required for token-level compression. "
                "Install it with: pip install scout-ai[llmlingua]"
            ) from None

        self._compressor = PromptCompressor(device_map=self._device)
        return self._compressor

    def compress(self, text: str, *, target_ratio: float = 0.5) -> CompressedContext:
        """Compress text using LLMLingua-2 token-level pruning."""
        if not text:
            return CompressedContext(
                text="",
                original_length=0,
                compressed_length=0,
                compression_ratio=1.0,
                method="llmlingua",
            )

        ratio = target_ratio or self._target_ratio
        compressor = self._get_compressor()

        # LLMLingua compress_prompt returns a dict with "compressed_prompt"
        result = compressor.compress_prompt(  # type: ignore[union-attr]
            text,
            rate=ratio,
            force_tokens=["\n", ".", "?", "!"],
        )

        compressed_text = result.get("compressed_prompt", text) if isinstance(result, dict) else text
        return CompressedContext(
            text=compressed_text,
            original_length=len(text),
            compressed_length=len(compressed_text),
            compression_ratio=len(compressed_text) / len(text) if text else 1.0,
            method="llmlingua",
            metadata={"device": self._device},
        )
