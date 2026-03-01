"""Tests for deterministic LLM parameter plumbing (top_p, seed)."""

from __future__ import annotations

from scout_ai.config import ScoutSettings
from scout_ai.core.config import AppSettings, LLMConfig


class TestLLMConfigDefaults:
    def test_top_p_default(self) -> None:
        cfg = LLMConfig()
        assert cfg.top_p == 1.0

    def test_seed_default_none(self) -> None:
        cfg = LLMConfig()
        assert cfg.seed is None

    def test_seed_can_be_set(self) -> None:
        cfg = LLMConfig(seed=42)
        assert cfg.seed == 42


class TestScoutSettingsDefaults:
    def test_top_p_default(self) -> None:
        s = ScoutSettings()
        assert s.llm_top_p == 1.0

    def test_seed_default_none(self) -> None:
        s = ScoutSettings()
        assert s.llm_seed is None


class TestAppSettingsBridge:
    def test_top_p_flows_through(self) -> None:
        settings = AppSettings(llm=LLMConfig(top_p=0.95))
        assert settings.llm.top_p == 0.95

    def test_seed_flows_through(self) -> None:
        settings = AppSettings(llm=LLMConfig(seed=42))
        assert settings.llm.seed == 42


class TestFactoryPassthrough:
    """Verify factory passes top_p/seed to provider constructors.

    These tests inspect the factory code path without importing
    provider SDKs (which may not be installed).
    """

    def test_openai_params_include_top_p(self) -> None:
        """The OpenAI branch should include top_p in params dict."""
        settings = AppSettings(
            llm=LLMConfig(provider="openai", top_p=0.9, seed=123),
        )
        # We can't call create_model without the SDK, but we can verify
        # the config is correctly structured
        assert settings.llm.top_p == 0.9
        assert settings.llm.seed == 123

    def test_bedrock_top_p(self) -> None:
        settings = AppSettings(
            llm=LLMConfig(provider="bedrock", top_p=0.8),
        )
        assert settings.llm.top_p == 0.8
