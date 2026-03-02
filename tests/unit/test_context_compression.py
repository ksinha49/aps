"""Tests for context compression backends."""

from __future__ import annotations

import pytest

from scout_ai.context.compression.entropic import EntropicCompressor
from scout_ai.context.compression.noop import NoOpCompressor
from scout_ai.context.models import CompressedContext


class TestNoOpCompressor:
    """NoOp compressor should pass through text unchanged."""

    def test_passthrough(self) -> None:
        c = NoOpCompressor()
        result = c.compress("Hello world")
        assert result.text == "Hello world"
        assert result.compression_ratio == 1.0
        assert result.method == "noop"

    def test_empty_text(self) -> None:
        c = NoOpCompressor()
        result = c.compress("")
        assert result.text == ""
        assert result.compression_ratio == 1.0

    def test_returns_compressed_context(self) -> None:
        c = NoOpCompressor()
        result = c.compress("test")
        assert isinstance(result, CompressedContext)
        assert result.original_length == 4
        assert result.compressed_length == 4


class TestEntropicCompressor:
    """Entropic compressor should reduce text by removing low-entropy sentences."""

    def test_short_text_unchanged(self) -> None:
        """Text below min_tokens should not be compressed."""
        c = EntropicCompressor(min_tokens=500)
        short = "This is a short sentence."
        result = c.compress(short)
        assert result.text == short
        assert result.compression_ratio == 1.0
        assert result.metadata.get("skipped") is True

    def test_reduces_long_text(self) -> None:
        """Long repetitive text should be compressed."""
        c = EntropicCompressor(min_tokens=10)
        # Build text with boilerplate repetition
        unique = "The patient has hypertension and takes metoprolol daily."
        boilerplate = "Please see the attached document for details."
        text = " ".join([unique] + [boilerplate] * 20 + [unique])
        result = c.compress(text, target_ratio=0.3)
        assert result.compressed_length < result.original_length
        assert result.compression_ratio < 1.0
        assert result.method == "entropic"

    def test_empty_text(self) -> None:
        c = EntropicCompressor()
        result = c.compress("")
        assert result.text == ""
        assert result.compression_ratio == 1.0

    def test_single_sentence(self) -> None:
        c = EntropicCompressor(min_tokens=0)
        result = c.compress("One sentence only.")
        assert result.text == "One sentence only."
        assert result.compression_ratio == 1.0

    def test_preserves_original_order(self) -> None:
        """Kept sentences should maintain their original order."""
        c = EntropicCompressor(min_tokens=0)
        text = "First unique info. " + " ".join(["Filler text." for _ in range(20)]) + " Last unique info."
        result = c.compress(text, target_ratio=0.3)
        if "First" in result.text and "Last" in result.text:
            assert result.text.index("First") < result.text.index("Last")


class TestLLMLinguaImportError:
    """LLMLingua compressor should give a clear error when not installed."""

    def test_import_error_message(self) -> None:
        from scout_ai.context.compression.llmlingua import LLMLinguaCompressor

        c = LLMLinguaCompressor()
        with pytest.raises(ImportError, match="pip install scout-ai"):
            c.compress("test text")


class TestCompressionFactory:
    """create_compressor() factory should dispatch correctly."""

    def test_default_returns_noop(self) -> None:
        from scout_ai.context.compression import create_compressor

        c = create_compressor(None)
        assert isinstance(c, NoOpCompressor)

    def test_noop_method(self) -> None:
        from scout_ai.context.compression import create_compressor
        from scout_ai.core.config import CompressionConfig

        config = CompressionConfig(enabled=True, method="noop")

        class FakeSettings:
            compression = config

        c = create_compressor(FakeSettings())
        assert isinstance(c, NoOpCompressor)

    def test_entropic_method(self) -> None:
        from scout_ai.context.compression import create_compressor
        from scout_ai.core.config import CompressionConfig

        config = CompressionConfig(enabled=True, method="entropic")

        class FakeSettings:
            compression = config

        c = create_compressor(FakeSettings())
        assert isinstance(c, EntropicCompressor)

    def test_disabled_returns_noop(self) -> None:
        from scout_ai.context.compression import create_compressor
        from scout_ai.core.config import CompressionConfig

        config = CompressionConfig(enabled=False, method="entropic")

        class FakeSettings:
            compression = config

        c = create_compressor(FakeSettings())
        assert isinstance(c, NoOpCompressor)
