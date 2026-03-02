"""NoOp compressor — passthrough with no transformation."""

from __future__ import annotations

from scout_ai.context.models import CompressedContext


class NoOpCompressor:
    """Passthrough compressor that returns input unchanged.

    Used as the default when compression is disabled.
    """

    def compress(self, text: str, *, target_ratio: float = 0.5) -> CompressedContext:
        """Return text unchanged with compression_ratio=1.0."""
        return CompressedContext(
            text=text,
            original_length=len(text),
            compressed_length=len(text),
            compression_ratio=1.0,
            method="noop",
        )
