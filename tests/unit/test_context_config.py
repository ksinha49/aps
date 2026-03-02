"""Tests for context engineering config classes — defaults and env overrides."""

from __future__ import annotations

from scout_ai.core.config import (
    CachingConfig,
    CompressionConfig,
    ContextCacheConfig,
    PrefixConfig,
)


class TestCompressionConfig:
    """CompressionConfig should have correct defaults and accept env vars."""

    def test_defaults(self) -> None:
        config = CompressionConfig()
        assert config.enabled is False
        assert config.method == "noop"
        assert config.target_ratio == 0.5
        assert config.min_tokens_for_compression == 500

    def test_env_override(self, monkeypatch: object) -> None:
        import pytest

        mp = pytest.MonkeyPatch()
        mp.setenv("SCOUT_COMPRESSION_ENABLED", "true")
        mp.setenv("SCOUT_COMPRESSION_METHOD", "entropic")
        mp.setenv("SCOUT_COMPRESSION_TARGET_RATIO", "0.3")
        try:
            config = CompressionConfig()
            assert config.enabled is True
            assert config.method == "entropic"
            assert config.target_ratio == 0.3
        finally:
            mp.undo()


class TestPrefixConfig:
    """PrefixConfig should have correct defaults."""

    def test_defaults(self) -> None:
        config = PrefixConfig()
        assert config.enabled is False
        assert config.sort_strategy == "page_number"
        assert config.deterministic_json is True

    def test_env_override(self, monkeypatch: object) -> None:
        import pytest

        mp = pytest.MonkeyPatch()
        mp.setenv("SCOUT_PREFIX_ENABLED", "true")
        mp.setenv("SCOUT_PREFIX_SORT_STRATEGY", "section_path")
        try:
            config = PrefixConfig()
            assert config.enabled is True
            assert config.sort_strategy == "section_path"
        finally:
            mp.undo()


class TestContextCacheConfig:
    """ContextCacheConfig should have correct defaults."""

    def test_defaults(self) -> None:
        config = ContextCacheConfig()
        assert config.enabled is False
        assert config.backend == "memory"
        assert config.ttl_seconds == 3600
        assert config.max_entries == 1000
        assert config.l1_max_size == 100
        assert config.redis_url == ""

    def test_env_override(self, monkeypatch: object) -> None:
        import pytest

        mp = pytest.MonkeyPatch()
        mp.setenv("SCOUT_CONTEXT_CACHE_ENABLED", "true")
        mp.setenv("SCOUT_CONTEXT_CACHE_BACKEND", "redis")
        mp.setenv("SCOUT_CONTEXT_CACHE_TTL_SECONDS", "7200")
        try:
            config = ContextCacheConfig()
            assert config.enabled is True
            assert config.backend == "redis"
            assert config.ttl_seconds == 7200
        finally:
            mp.undo()


class TestCachingConfigExtensions:
    """CachingConfig should have new multi-breakpoint fields."""

    def test_new_fields_defaults(self) -> None:
        config = CachingConfig()
        assert config.max_breakpoints == 4
        assert config.cache_document_layer is True
        assert config.cache_tool_layer is False
