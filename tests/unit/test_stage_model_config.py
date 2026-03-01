"""Tests for per-stage model configuration."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from scout_ai.core.config import AppSettings, LLMConfig, StageModelConfig


class TestStageModelConfig:
    """StageModelConfig defaults and field resolution."""

    def test_defaults_to_empty_strings(self) -> None:
        config = StageModelConfig()
        assert config.indexing_model == ""
        assert config.retrieval_model == ""
        assert config.extraction_model == ""
        assert config.synthesis_model == ""

    def test_stage_override_takes_precedence(self) -> None:
        settings = AppSettings(
            stage_models=StageModelConfig(retrieval_model="claude-3-haiku-20240307"),
        )
        effective = settings.stage_models.retrieval_model or settings.llm.model
        assert effective == "claude-3-haiku-20240307"

    def test_empty_stage_falls_back_to_default(self) -> None:
        settings = AppSettings(
            llm=LLMConfig(model="gpt-4o"),
            stage_models=StageModelConfig(),
        )
        effective = settings.stage_models.retrieval_model or settings.llm.model
        assert effective == "gpt-4o"

    def test_each_stage_independently_configurable(self) -> None:
        settings = AppSettings(
            llm=LLMConfig(model="default-model"),
            stage_models=StageModelConfig(
                indexing_model="fast-model",
                extraction_model="smart-model",
            ),
        )
        assert (settings.stage_models.indexing_model or settings.llm.model) == "fast-model"
        assert (settings.stage_models.retrieval_model or settings.llm.model) == "default-model"
        assert (settings.stage_models.extraction_model or settings.llm.model) == "smart-model"
        assert (settings.stage_models.synthesis_model or settings.llm.model) == "default-model"

    def test_stage_models_accessible_from_app_settings(self) -> None:
        """StageModelConfig is nested in AppSettings as ``stage_models``."""
        settings = AppSettings()
        assert isinstance(settings.stage_models, StageModelConfig)


class TestCreateModelOverride:
    """create_model respects the model_override parameter."""

    def test_override_used_for_ollama(self) -> None:
        """When model_override is non-empty, OllamaModel receives it instead of settings.llm.model."""
        settings = AppSettings(llm=LLMConfig(provider="ollama", model="default-model"))
        mock_ollama = MagicMock()

        with patch("scout_ai.agents.factory.OllamaModel", mock_ollama, create=True):
            # We need to actually import after patching.  Simplest: import the
            # function here so the lazy import inside hits our mock.
            with patch.dict("sys.modules", {"strands.models.ollama": MagicMock(OllamaModel=mock_ollama)}):
                from scout_ai.agents.factory import create_model

                create_model(settings, model_override="custom-override")

        mock_ollama.assert_called_once()
        call_kwargs = mock_ollama.call_args
        assert call_kwargs[1]["model_id"] == "custom-override"

    def test_empty_override_falls_back(self) -> None:
        """When model_override is empty, settings.llm.model is used."""
        settings = AppSettings(llm=LLMConfig(provider="ollama", model="default-model"))
        mock_ollama = MagicMock()

        with patch.dict("sys.modules", {"strands.models.ollama": MagicMock(OllamaModel=mock_ollama)}):
            with patch("scout_ai.agents.factory.OllamaModel", mock_ollama, create=True):
                from scout_ai.agents.factory import create_model

                create_model(settings, model_override="")

        mock_ollama.assert_called_once()
        call_kwargs = mock_ollama.call_args
        assert call_kwargs[1]["model_id"] == "default-model"

    def test_override_used_for_openai(self) -> None:
        """OpenAI provider branch also respects model_override."""
        settings = AppSettings(llm=LLMConfig(provider="openai", model="gpt-4o", api_key="test-key"))
        mock_openai = MagicMock()

        with patch.dict("sys.modules", {"strands.models.openai": MagicMock(OpenAIModel=mock_openai)}):
            with patch("scout_ai.agents.factory.OpenAIModel", mock_openai, create=True):
                from scout_ai.agents.factory import create_model

                create_model(settings, model_override="gpt-4o-mini")

        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args
        assert call_kwargs[1]["model_id"] == "gpt-4o-mini"

    def test_override_used_for_anthropic(self) -> None:
        """Anthropic provider branch prepends 'anthropic/' to the effective model."""
        settings = AppSettings(llm=LLMConfig(provider="anthropic", model="claude-3-opus"))
        mock_litellm = MagicMock()

        with patch.dict("sys.modules", {"strands.models.litellm": MagicMock(LiteLLMModel=mock_litellm)}):
            with patch("scout_ai.agents.factory.LiteLLMModel", mock_litellm, create=True):
                from scout_ai.agents.factory import create_model

                create_model(settings, model_override="claude-3-haiku")

        mock_litellm.assert_called_once()
        call_kwargs = mock_litellm.call_args
        assert call_kwargs[1]["model_id"] == "anthropic/claude-3-haiku"

    def test_override_used_for_litellm(self) -> None:
        """LiteLLM provider branch uses effective_model directly."""
        settings = AppSettings(llm=LLMConfig(provider="litellm", model="default-model"))
        mock_litellm = MagicMock()

        with patch.dict("sys.modules", {"strands.models.litellm": MagicMock(LiteLLMModel=mock_litellm)}):
            with patch("scout_ai.agents.factory.LiteLLMModel", mock_litellm, create=True):
                from scout_ai.agents.factory import create_model

                create_model(settings, model_override="overridden-model")

        mock_litellm.assert_called_once()
        call_kwargs = mock_litellm.call_args
        assert call_kwargs[1]["model_id"] == "overridden-model"
