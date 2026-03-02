"""Integration test: full extraction pipeline with compression enabled."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scout_ai.context.compression.entropic import EntropicCompressor
from scout_ai.context.compression.noop import NoOpCompressor


def _mock_response(content: str) -> MagicMock:
    """Build a mock LiteLLM response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    return response


class TestCompressedExtractionPipeline:
    """Verify compression integrates correctly with extraction."""

    def test_noop_compressor_passthrough(self) -> None:
        """NoOp compressor should not change extraction context."""
        compressor = NoOpCompressor()
        context = "Patient has hypertension. Blood pressure 140/90."
        result = compressor.compress(context, target_ratio=0.5)
        assert result.text == context
        assert result.compression_ratio == 1.0

    def test_entropic_compressor_reduces_context(self) -> None:
        """Entropic compressor should reduce long repetitive context."""
        compressor = EntropicCompressor(min_tokens=10)
        # Simulate a real context with repeated boilerplate
        unique = "The patient was diagnosed with type 2 diabetes mellitus."
        boilerplate = "Please refer to the attached documentation for additional information."
        context = " ".join([unique] + [boilerplate] * 30 + [unique])

        result = compressor.compress(context, target_ratio=0.3)
        assert result.compressed_length < result.original_length
        assert result.method == "entropic"

    @pytest.mark.asyncio()
    async def test_chat_with_compression(self) -> None:
        """ScoutChat should use compressor instead of hard truncation."""
        from scout_ai.config import ScoutSettings
        from scout_ai.models import ExtractionQuestion
        from scout_ai.providers.pageindex.chat import ScoutChat
        from scout_ai.providers.pageindex.client import LLMClient

        mock_llm_response = {
            "answers": [
                {
                    "question_id": "q1",
                    "answer": "Hypertension",
                    "confidence": 0.95,
                    "citations": [{"page_number": 1, "verbatim_quote": "Patient has hypertension"}],
                }
            ]
        }

        async def _mock_acomp(**kwargs: Any) -> MagicMock:
            return _mock_response(json.dumps(mock_llm_response))

        settings = ScoutSettings()
        client = LLMClient(settings)
        compressor = NoOpCompressor()
        chat = ScoutChat(settings, client, compressor=compressor, max_context_chars=8000)

        questions = [
            ExtractionQuestion(
                question_id="q1",
                question_text="What conditions does the patient have?",
                category="medical_history",
                tier=1,
            )
        ]
        with patch("litellm.acompletion", side_effect=_mock_acomp):
            results = await chat.extract_answers(questions, "Patient has hypertension.")
        assert len(results) == 1
        assert results[0].question_id == "q1"
