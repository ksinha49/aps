"""Tests for LLMClient with mocked litellm.acompletion — system_prompt + cache_control formatting."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pageindex_rag.config import PageIndexSettings
from pageindex_rag.providers.pageindex.client import LLMClient


def _make_settings(**overrides: Any) -> PageIndexSettings:
    defaults = {
        "llm_base_url": "http://localhost:4000/v1",
        "llm_api_key": "test-key",
        "llm_model": "anthropic/claude-sonnet-4-20250514",
        "llm_temperature": 0.0,
        "llm_max_retries": 1,
    }
    defaults.update(overrides)
    return PageIndexSettings(**defaults)


def _mock_response(content: str = "test response") -> MagicMock:
    """Build a mock LiteLLM response object."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    return response


class TestLLMClientCaching:
    @pytest.mark.asyncio
    async def test_complete_without_system_prompt(self) -> None:
        """Default call: no system prompt, no cache_control."""
        client = LLMClient(_make_settings())

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = _mock_response("hello")
            result = await client.complete("test prompt")

        assert result == "hello"
        call_kwargs = mock_acomp.call_args
        messages = call_kwargs.kwargs["messages"]

        # Only user message, no system message
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "test prompt"

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt_no_cache(self) -> None:
        """System prompt present but caching disabled — no cache_control."""
        client = LLMClient(_make_settings())

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = _mock_response("answer")
            result = await client.complete(
                "question",
                system_prompt="You are a doctor.",
                cache_system=False,
            )

        assert result == "answer"
        messages = mock_acomp.call_args.kwargs["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        # Content is a list of content blocks
        assert isinstance(messages[0]["content"], list)
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][0]["text"] == "You are a doctor."
        # No cache_control when cache_system=False
        assert "cache_control" not in messages[0]["content"][0]

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt_and_cache(self) -> None:
        """System prompt with cache_system=True adds cache_control."""
        client = LLMClient(_make_settings())

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = _mock_response("cached answer")
            result = await client.complete(
                "question",
                system_prompt="Document context here...",
                cache_system=True,
            )

        assert result == "cached answer"
        messages = mock_acomp.call_args.kwargs["messages"]

        assert len(messages) == 2
        system_content = messages[0]["content"][0]
        assert system_content["cache_control"] == {"type": "ephemeral"}
        assert system_content["text"] == "Document context here..."

    @pytest.mark.asyncio
    async def test_complete_with_chat_history(self) -> None:
        """Chat history is placed between system and user messages."""
        client = LLMClient(_make_settings())
        history = [
            {"role": "user", "content": "prior question"},
            {"role": "assistant", "content": "prior answer"},
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = _mock_response("followup")
            await client.complete(
                "new question",
                system_prompt="System",
                cache_system=True,
                chat_history=history,
            )

        messages = mock_acomp.call_args.kwargs["messages"]
        assert len(messages) == 4  # system + 2 history + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "prior question"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "new question"

    @pytest.mark.asyncio
    async def test_complete_batch_with_caching(self) -> None:
        """Batch completion forwards system_prompt and cache_system."""
        client = LLMClient(_make_settings())

        call_count = 0

        async def _mock_acomp(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            return _mock_response(f"answer-{call_count}")

        with patch("litellm.acompletion", side_effect=_mock_acomp):
            results = await client.complete_batch(
                ["q1", "q2", "q3"],
                system_prompt="Cached context",
                cache_system=True,
            )

        assert len(results) == 3
        assert all(r.startswith("answer-") for r in results)

    @pytest.mark.asyncio
    async def test_finish_reason_mapping(self) -> None:
        """Finish reason 'length' maps to 'max_output_reached'."""
        client = LLMClient(_make_settings())
        resp = _mock_response("truncated")
        resp.choices[0].finish_reason = "length"

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = resp
            content, reason = await client.complete_with_finish_reason("prompt")

        assert content == "truncated"
        assert reason == "max_output_reached"

    @pytest.mark.asyncio
    async def test_model_override(self) -> None:
        """Custom model is passed through to acompletion."""
        client = LLMClient(_make_settings())

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = _mock_response("ok")
            await client.complete("prompt", model="bedrock/anthropic.claude-v2")

        assert mock_acomp.call_args.kwargs["model"] == "bedrock/anthropic.claude-v2"
