"""Context compression: factory + backend implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scout_ai.context.compression.noop import NoOpCompressor
from scout_ai.context.protocols import IContextCompressor

if TYPE_CHECKING:
    from scout_ai.core.config import CompressionConfig

__all__ = [
    "create_compressor",
    "NoOpCompressor",
]


def create_compressor(settings: object | None = None) -> IContextCompressor:
    """Create a compressor from settings.

    Args:
        settings: An ``AppSettings`` or ``CompressionConfig`` instance.
            If None, returns NoOpCompressor.
    """
    config: CompressionConfig | None = None

    if settings is not None:
        # Accept either AppSettings (has .compression) or CompressionConfig directly
        config = getattr(settings, "compression", None)
        if config is None and hasattr(settings, "method"):
            config = settings  # type: ignore[assignment]

    if config is None or not config.enabled:
        return NoOpCompressor()

    method = config.method
    if method == "noop":
        return NoOpCompressor()
    elif method == "entropic":
        from scout_ai.context.compression.entropic import EntropicCompressor

        return EntropicCompressor(min_tokens=config.min_tokens_for_compression)
    elif method == "llmlingua":
        from scout_ai.context.compression.llmlingua import LLMLinguaCompressor

        return LLMLinguaCompressor(target_ratio=config.target_ratio)
    else:
        raise ValueError(f"Unknown compression method: {method!r}")
