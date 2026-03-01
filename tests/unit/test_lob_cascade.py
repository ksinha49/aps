"""Tests for LOB and region cascade from top-level settings to prompt/rules defaults."""

from __future__ import annotations

from scout_ai.core.config import AppSettings, PromptConfig


class TestLobCascade:
    """Verify LOB cascade logic used in app.py lifespan."""

    def test_lob_cascades_when_prompt_default_is_wildcard(self) -> None:
        """When prompt.default_lob is '*', top-level lob should win."""
        settings = AppSettings(lob="retirement_plans")
        effective = settings.prompt.default_lob if settings.prompt.default_lob != "*" else settings.lob
        assert effective == "retirement_plans"

    def test_prompt_lob_overrides_when_explicitly_set(self) -> None:
        """When prompt.default_lob is explicitly set, it wins over top-level."""
        settings = AppSettings(
            lob="retirement_plans",
            prompt=PromptConfig(default_lob="group_health"),
        )
        effective = settings.prompt.default_lob if settings.prompt.default_lob != "*" else settings.lob
        assert effective == "group_health"

    def test_prompt_region_cascades_from_top_level(self) -> None:
        """When prompt.aws_region is empty, top-level aws_region fills in."""
        settings = AppSettings(aws_region="eu-west-1", prompt=PromptConfig(aws_region=""))
        effective = settings.prompt.aws_region or settings.aws_region
        assert effective == "eu-west-1"

    def test_prompt_region_overrides_top_level(self) -> None:
        """When prompt.aws_region is explicitly set, it wins."""
        settings = AppSettings(
            aws_region="eu-west-1",
            prompt=PromptConfig(aws_region="us-east-1"),
        )
        effective = settings.prompt.aws_region or settings.aws_region
        assert effective == "us-east-1"
