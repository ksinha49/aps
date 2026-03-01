"""Tests for startup validation checks."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from scout_ai.core.config import AppSettings, AuthConfig, LLMConfig, PersistenceConfig


class TestApiKeyValidation:
    """Validate that placeholder API keys are rejected for providers that need them."""

    def test_rejects_no_key_for_openai(self):
        settings = AppSettings(llm=LLMConfig(provider="openai", api_key="no-key"))
        with pytest.raises(ValueError, match="SCOUT_LLM_API_KEY is required"):
            from scout_ai.core.startup_checks import validate_settings

            validate_settings(settings)

    def test_rejects_empty_key_for_openai(self):
        settings = AppSettings(llm=LLMConfig(provider="openai", api_key=""))
        with pytest.raises(ValueError, match="SCOUT_LLM_API_KEY is required"):
            from scout_ai.core.startup_checks import validate_settings

            validate_settings(settings)

    def test_rejects_no_key_for_anthropic(self):
        settings = AppSettings(llm=LLMConfig(provider="anthropic", api_key="no-key"))
        with pytest.raises(ValueError, match="SCOUT_LLM_API_KEY is required"):
            from scout_ai.core.startup_checks import validate_settings

            validate_settings(settings)

    def test_rejects_no_key_for_litellm(self):
        settings = AppSettings(llm=LLMConfig(provider="litellm", api_key="no-key"))
        with pytest.raises(ValueError, match="SCOUT_LLM_API_KEY is required"):
            from scout_ai.core.startup_checks import validate_settings

            validate_settings(settings)

    def test_accepts_no_key_for_bedrock(self):
        """Bedrock uses IAM roles, no API key needed."""
        settings = AppSettings(llm=LLMConfig(provider="bedrock", api_key="no-key"))
        from scout_ai.core.startup_checks import validate_settings

        validate_settings(settings)  # Should not raise

    def test_accepts_no_key_for_ollama(self):
        """Ollama is local, no API key needed."""
        settings = AppSettings(llm=LLMConfig(provider="ollama", api_key="no-key"))
        from scout_ai.core.startup_checks import validate_settings

        validate_settings(settings)  # Should not raise

    def test_accepts_real_key_for_openai(self):
        settings = AppSettings(llm=LLMConfig(provider="openai", api_key="sk-real-key-here"))
        from scout_ai.core.startup_checks import validate_settings

        validate_settings(settings)  # Should not raise


class TestPersistenceCheck:
    """Validate container persistence warnings."""

    def test_warns_file_backend_in_ecs(self):
        settings = AppSettings(persistence=PersistenceConfig(backend="file"))
        with patch.dict(os.environ, {"ECS_CONTAINER_METADATA_URI": "http://169.254.170.2/v4"}):
            from scout_ai.core.startup_checks import validate_settings

            with patch("scout_ai.core.startup_checks.log") as mock_log:
                validate_settings(settings)
                mock_log.warning.assert_called()

    def test_warns_file_backend_in_k8s(self):
        settings = AppSettings(persistence=PersistenceConfig(backend="file"))
        with patch.dict(os.environ, {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}):
            from scout_ai.core.startup_checks import validate_settings

            with patch("scout_ai.core.startup_checks.log") as mock_log:
                validate_settings(settings)
                mock_log.warning.assert_called()

    def test_no_warning_for_s3_backend(self):
        settings = AppSettings(
            llm=LLMConfig(provider="bedrock"),
            persistence=PersistenceConfig(backend="s3"),
        )
        with patch.dict(os.environ, {"ECS_CONTAINER_METADATA_URI": "http://169.254.170.2/v4"}):
            from scout_ai.core.startup_checks import validate_settings

            with patch("scout_ai.core.startup_checks.log") as mock_log:
                validate_settings(settings)
                mock_log.warning.assert_not_called()


class TestAuthCheck:
    """Validate auth configuration warnings."""

    def test_warns_auth_enabled_no_credentials(self):
        settings = AppSettings(
            llm=LLMConfig(provider="bedrock"),
            auth=AuthConfig(enabled=True, api_keys=[], jwks_url=""),
        )
        from scout_ai.core.startup_checks import validate_settings

        with patch("scout_ai.core.startup_checks.log") as mock_log:
            validate_settings(settings)
            mock_log.warning.assert_called()
