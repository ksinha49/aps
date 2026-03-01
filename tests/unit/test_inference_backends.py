"""Tests for the pluggable inference backend layer."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scout_ai.core.config import AppSettings, LLMConfig
from scout_ai.inference.factory import create_inference_backend
from scout_ai.inference.protocols import (
    IInferenceBackend,
    InferenceRequest,
    InferenceResult,
)
from scout_ai.inference.realtime import RealTimeBackend
from tests.fakes.fake_inference import FakeInferenceBackend

# ── Protocol compliance ──────────────────────────────────────────────


class TestProtocolCompliance:
    def test_fake_backend_satisfies_protocol(self) -> None:
        backend = FakeInferenceBackend()
        assert isinstance(backend, IInferenceBackend)

    def test_realtime_backend_satisfies_protocol(self) -> None:
        backend = RealTimeBackend()
        assert isinstance(backend, IInferenceBackend)


# ── InferenceRequest / InferenceResult dataclasses ───────────────────


class TestDataModels:
    def test_inference_request_defaults(self) -> None:
        req = InferenceRequest(
            request_id="r1",
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o",
        )
        assert req.request_id == "r1"
        assert req.params == {}

    def test_inference_result_defaults(self) -> None:
        result = InferenceResult(content="hello")
        assert result.finish_reason == "finished"
        assert result.usage == {}
        assert result.request_id == ""

    def test_inference_request_with_params(self) -> None:
        req = InferenceRequest(
            request_id="r2",
            messages=[],
            model="claude-3",
            params={"temperature": 0.5},
        )
        assert req.params == {"temperature": 0.5}


# ── RealTimeBackend ──────────────────────────────────────────────────


class TestRealTimeBackend:
    @pytest.mark.asyncio
    async def test_infer_delegates_to_litellm(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            backend = RealTimeBackend()
            result = await backend.infer(
                [{"role": "user", "content": "hi"}],
                "gpt-4o",
                temperature=0.0,
            )

        assert result.content == "test response"
        assert result.finish_reason == "finished"
        assert result.usage["total_tokens"] == 15
        mock_acomp.assert_called_once()

    @pytest.mark.asyncio
    async def test_infer_maps_length_finish_reason(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "truncated"
        mock_response.choices[0].finish_reason = "length"
        mock_response.usage = None

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            backend = RealTimeBackend()
            result = await backend.infer([], "gpt-4o")

        assert result.finish_reason == "max_output_reached"
        assert result.usage == {}

    @pytest.mark.asyncio
    async def test_infer_batch_respects_concurrency(self) -> None:
        call_count = 0
        max_concurrent = 0
        current_concurrent = 0

        async def tracking_infer(self: Any, messages: Any, model: Any, **params: Any) -> InferenceResult:
            nonlocal call_count, max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            call_count += 1
            await asyncio.sleep(0.01)
            current_concurrent -= 1
            return InferenceResult(content=f"resp-{call_count}")

        backend = RealTimeBackend(max_concurrent=2)
        requests = [
            InferenceRequest(request_id=str(i), messages=[], model="m")
            for i in range(5)
        ]

        with patch.object(RealTimeBackend, "infer", tracking_infer):
            results = await backend.infer_batch(requests)

        assert len(results) == 5
        assert max_concurrent <= 2


# ── Factory ──────────────────────────────────────────────────────────


class TestInferenceFactory:
    def test_default_returns_realtime_backend(self) -> None:
        settings = AppSettings()
        backend = create_inference_backend(settings)
        assert isinstance(backend, RealTimeBackend)

    def test_realtime_explicit(self) -> None:
        settings = AppSettings(llm=LLMConfig(inference_backend="realtime"))
        backend = create_inference_backend(settings)
        assert isinstance(backend, RealTimeBackend)

    def test_dotted_path_loads_external_class(self) -> None:
        # External backends receive settings as first positional arg.
        # FakeInferenceBackend uses **kwargs only, so we wrap it for this test.
        settings = AppSettings(
            llm=LLMConfig(inference_backend="tests.fakes.fake_inference:FakeInferenceBackend")
        )
        # Patch _import_dotted_path to return a lambda that ignores settings
        with patch(
            "scout_ai.domains.registry._import_dotted_path",
            return_value=lambda _settings: FakeInferenceBackend(),
        ):
            backend = create_inference_backend(settings)
        assert isinstance(backend, FakeInferenceBackend)

    def test_dotted_path_not_found_raises(self) -> None:
        settings = AppSettings(
            llm=LLMConfig(inference_backend="nonexistent.module:Missing")
        )
        with pytest.raises(ImportError):
            create_inference_backend(settings)

    def test_dotted_path_not_callable_raises(self) -> None:
        settings = AppSettings(
            llm=LLMConfig(inference_backend="some.module:NotCallable")
        )
        with patch(
            "scout_ai.domains.registry._import_dotted_path",
            return_value="not_callable_string",
        ):
            with pytest.raises(TypeError, match="not callable"):
                create_inference_backend(settings)


# ── FakeInferenceBackend ─────────────────────────────────────────────


class TestFakeInferenceBackend:
    @pytest.mark.asyncio
    async def test_infer_returns_canned_response(self) -> None:
        backend = FakeInferenceBackend(default_content="canned")
        result = await backend.infer(
            [{"role": "user", "content": "hello"}], "model-1"
        )
        assert result.content == "canned"
        assert result.finish_reason == "finished"

    @pytest.mark.asyncio
    async def test_infer_records_calls(self) -> None:
        backend = FakeInferenceBackend()
        await backend.infer([{"role": "user", "content": "q1"}], "m1")
        await backend.infer([{"role": "user", "content": "q2"}], "m2")
        assert len(backend.calls) == 2
        assert backend.calls[0]["model"] == "m1"
        assert backend.calls[1]["model"] == "m2"

    @pytest.mark.asyncio
    async def test_infer_batch_preserves_request_ids(self) -> None:
        backend = FakeInferenceBackend()
        requests = [
            InferenceRequest(request_id="a", messages=[], model="m"),
            InferenceRequest(request_id="b", messages=[], model="m"),
        ]
        results = await backend.infer_batch(requests)
        assert [r.request_id for r in results] == ["a", "b"]


# ── LLMClient delegation ────────────────────────────────────────────


class TestLLMClientBackendDelegation:
    @pytest.mark.asyncio
    async def test_complete_delegates_to_backend(self) -> None:
        from scout_ai.config import ScoutSettings
        from scout_ai.providers.pageindex.client import LLMClient

        backend = FakeInferenceBackend(default_content="backend result")
        settings = ScoutSettings()
        client = LLMClient(settings, backend=backend)

        result = await client.complete("hello")
        assert result == "backend result"
        assert len(backend.calls) == 1

    @pytest.mark.asyncio
    async def test_complete_with_finish_reason_delegates(self) -> None:
        from scout_ai.config import ScoutSettings
        from scout_ai.providers.pageindex.client import LLMClient

        backend = FakeInferenceBackend(
            default_content="done", default_finish_reason="max_output_reached"
        )
        settings = ScoutSettings()
        client = LLMClient(settings, backend=backend)

        content, reason = await client.complete_with_finish_reason("hello")
        assert content == "done"
        assert reason == "max_output_reached"

    @pytest.mark.asyncio
    async def test_complete_batch_delegates_to_backend(self) -> None:
        from scout_ai.config import ScoutSettings
        from scout_ai.providers.pageindex.client import LLMClient

        backend = FakeInferenceBackend(default_content="batch result")
        settings = ScoutSettings()
        client = LLMClient(settings, backend=backend)

        results = await client.complete_batch(["q1", "q2", "q3"])
        assert results == ["batch result", "batch result", "batch result"]
        assert len(backend.calls) == 3

    @pytest.mark.asyncio
    async def test_none_backend_uses_inline_path(self) -> None:
        """When backend=None, LLMClient uses the inline litellm path."""
        from scout_ai.config import ScoutSettings
        from scout_ai.providers.pageindex.client import LLMClient

        settings = ScoutSettings()
        client = LLMClient(settings, backend=None)
        assert client._backend is None

    def test_build_messages_basic(self) -> None:
        from scout_ai.config import ScoutSettings
        from scout_ai.providers.pageindex.client import LLMClient

        client = LLMClient(ScoutSettings())
        msgs = client._build_messages("hello")
        assert len(msgs) == 1
        assert msgs[0] == {"role": "user", "content": "hello"}

    def test_build_messages_with_system(self) -> None:
        from scout_ai.config import ScoutSettings
        from scout_ai.providers.pageindex.client import LLMClient

        client = LLMClient(ScoutSettings())
        msgs = client._build_messages("q", system_prompt="sys")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1] == {"role": "user", "content": "q"}

    def test_build_messages_with_cache_control(self) -> None:
        from scout_ai.config import ScoutSettings
        from scout_ai.providers.pageindex.client import LLMClient

        client = LLMClient(ScoutSettings())
        msgs = client._build_messages("q", system_prompt="sys", cache_system=True)
        sys_content = msgs[0]["content"]
        assert sys_content[0]["cache_control"] == {"type": "ephemeral"}

    def test_build_messages_with_chat_history(self) -> None:
        from scout_ai.config import ScoutSettings
        from scout_ai.providers.pageindex.client import LLMClient

        history = [{"role": "user", "content": "prev"}, {"role": "assistant", "content": "resp"}]
        client = LLMClient(ScoutSettings())
        msgs = client._build_messages("q", chat_history=history)
        assert len(msgs) == 3
        assert msgs[0]["content"] == "prev"
        assert msgs[2] == {"role": "user", "content": "q"}


# ── Config defaults ──────────────────────────────────────────────────


class TestInferenceConfig:
    def test_default_is_realtime(self) -> None:
        cfg = LLMConfig()
        assert cfg.inference_backend == "realtime"

    def test_app_settings_default(self) -> None:
        settings = AppSettings()
        assert settings.llm.inference_backend == "realtime"
