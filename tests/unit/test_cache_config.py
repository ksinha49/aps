"""Tests for CachingConfig defaults and env var overrides."""

from __future__ import annotations

import pytest

from scout_ai.core.config import AppSettings, CachingConfig, LLMConfig


class TestCachingConfig:
    def test_defaults(self) -> None:
        cfg = CachingConfig()
        assert cfg.enabled is False
        assert cfg.cache_system_prompt is True
        assert cfg.cache_document_context is True
        assert cfg.min_cacheable_tokens == 1024
        assert cfg.keepalive_interval_seconds == 240.0
        assert cfg.ttl_type == "ephemeral"

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCOUT_CACHING_ENABLED", "true")
        monkeypatch.setenv("SCOUT_CACHING_MIN_CACHEABLE_TOKENS", "2048")
        monkeypatch.setenv("SCOUT_CACHING_TTL_TYPE", "long")

        cfg = CachingConfig()
        assert cfg.enabled is True
        assert cfg.min_cacheable_tokens == 2048
        assert cfg.ttl_type == "long"

    def test_app_settings_includes_caching(self) -> None:
        settings = AppSettings()
        assert hasattr(settings, "caching")
        assert isinstance(settings.caching, CachingConfig)
        assert settings.caching.enabled is False


class TestLLMConfigProviders:
    def test_anthropic_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCOUT_LLM_PROVIDER", "anthropic")
        cfg = LLMConfig()
        assert cfg.provider == "anthropic"

    def test_default_provider(self) -> None:
        cfg = LLMConfig()
        assert cfg.provider == "ollama"

    def test_all_providers_accepted(self) -> None:
        for provider in ("bedrock", "openai", "ollama", "litellm", "anthropic"):
            cfg = LLMConfig(provider=provider)  # type: ignore[call-arg]
            assert cfg.provider == provider
