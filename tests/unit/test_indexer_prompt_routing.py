"""Tests for domain-parameterized indexer prompts."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestDefaultPrompt:
    def test_uses_specified_domain(self) -> None:
        with patch("scout_ai.prompts.registry.get_prompt") as mock_get:
            mock_get.return_value = "test prompt"
            from scout_ai.providers.pageindex.indexer import _default_prompt

            result = _default_prompt("TOC_DETECT_PROMPT", domain="workers_comp")
            mock_get.assert_called_once_with("workers_comp", "indexing", "TOC_DETECT_PROMPT")
            assert result == "test prompt"

    def test_falls_back_to_base(self) -> None:
        with patch("scout_ai.prompts.registry.get_prompt") as mock_get:
            mock_get.side_effect = [KeyError("not found"), "base prompt"]
            from scout_ai.providers.pageindex.indexer import _default_prompt

            result = _default_prompt("TOC_DETECT_PROMPT", domain="unknown")
            assert result == "base prompt"
            assert mock_get.call_count == 2
            mock_get.assert_any_call("unknown", "indexing", "TOC_DETECT_PROMPT")
            mock_get.assert_any_call("base", "indexing", "TOC_DETECT_PROMPT")

    def test_default_domain_is_aps(self) -> None:
        with patch("scout_ai.prompts.registry.get_prompt") as mock_get:
            mock_get.return_value = "aps prompt"
            from scout_ai.providers.pageindex.indexer import _default_prompt

            result = _default_prompt("TOC_DETECT_PROMPT")
            mock_get.assert_called_once_with("aps", "indexing", "TOC_DETECT_PROMPT")
            assert result == "aps prompt"

    def test_all_prompt_names_resolve_via_indexing_category(self) -> None:
        """Every prompt name used by the indexer resolves through the 'indexing' category."""
        prompt_names = [
            "TOC_DETECT_PROMPT",
            "GENERATE_TOC_INIT_PROMPT",
            "GENERATE_TOC_CONTINUE_PROMPT",
            "CHECK_TITLE_APPEARANCE_PROMPT",
            "CHECK_TITLE_START_PROMPT",
            "GENERATE_SUMMARY_PROMPT",
            "GENERATE_DOC_DESCRIPTION_PROMPT",
        ]
        with patch("scout_ai.prompts.registry.get_prompt") as mock_get:
            mock_get.return_value = "resolved"
            from scout_ai.providers.pageindex.indexer import _default_prompt

            for name in prompt_names:
                mock_get.reset_mock()
                result = _default_prompt(name, domain="aps")
                mock_get.assert_called_once_with("aps", "indexing", name)
                assert result == "resolved"

    def test_base_fallback_raises_if_both_fail(self) -> None:
        """If neither domain nor base has the prompt, KeyError propagates."""
        with patch("scout_ai.prompts.registry.get_prompt") as mock_get:
            mock_get.side_effect = KeyError("not found")
            from scout_ai.providers.pageindex.indexer import _default_prompt

            with pytest.raises(KeyError):
                _default_prompt("NONEXISTENT_PROMPT", domain="unknown")
            assert mock_get.call_count == 2


class TestScoutIndexerDomain:
    def test_default_domain_is_aps(self) -> None:
        """ScoutIndexer defaults to 'aps' domain when not specified."""
        settings = MagicMock()
        settings.tokenizer_method = "simple"
        settings.tokenizer_model = None
        settings.enable_section_classification = False
        client = MagicMock()

        from scout_ai.providers.pageindex.indexer import ScoutIndexer

        indexer = ScoutIndexer(settings, client)
        assert indexer._domain == "aps"

    def test_custom_domain_stored(self) -> None:
        """ScoutIndexer stores the domain parameter."""
        settings = MagicMock()
        settings.tokenizer_method = "simple"
        settings.tokenizer_model = None
        settings.enable_section_classification = False
        client = MagicMock()

        from scout_ai.providers.pageindex.indexer import ScoutIndexer

        indexer = ScoutIndexer(settings, client, domain="workers_comp")
        assert indexer._domain == "workers_comp"

    def test_get_prompt_passes_domain(self) -> None:
        """_get_prompt delegates to _default_prompt with the instance domain."""
        settings = MagicMock()
        settings.tokenizer_method = "simple"
        settings.tokenizer_model = None
        settings.enable_section_classification = False
        client = MagicMock()

        from scout_ai.providers.pageindex.indexer import ScoutIndexer

        indexer = ScoutIndexer(settings, client, domain="disability")

        with patch("scout_ai.providers.pageindex.indexer._default_prompt") as mock_dp:
            mock_dp.return_value = "disability prompt"
            result = indexer._get_prompt("TOC_DETECT_PROMPT")
            mock_dp.assert_called_once_with("TOC_DETECT_PROMPT", domain="disability")
            assert result == "disability prompt"

    def test_get_prompt_caches_result(self) -> None:
        """_get_prompt caches the resolved prompt for subsequent calls."""
        settings = MagicMock()
        settings.tokenizer_method = "simple"
        settings.tokenizer_model = None
        settings.enable_section_classification = False
        client = MagicMock()

        from scout_ai.providers.pageindex.indexer import ScoutIndexer

        indexer = ScoutIndexer(settings, client, domain="aps")

        with patch("scout_ai.providers.pageindex.indexer._default_prompt") as mock_dp:
            mock_dp.return_value = "cached prompt"
            # First call resolves and caches
            result1 = indexer._get_prompt("TOC_DETECT_PROMPT")
            # Second call uses cache, no new call to _default_prompt
            result2 = indexer._get_prompt("TOC_DETECT_PROMPT")
            assert result1 == result2 == "cached prompt"
            mock_dp.assert_called_once()

    def test_injected_prompts_take_precedence(self) -> None:
        """Explicitly injected prompts override domain registry lookup."""
        settings = MagicMock()
        settings.tokenizer_method = "simple"
        settings.tokenizer_model = None
        settings.enable_section_classification = False
        client = MagicMock()

        from scout_ai.providers.pageindex.indexer import ScoutIndexer

        custom_prompts = {"TOC_DETECT_PROMPT": "custom injected prompt"}
        indexer = ScoutIndexer(settings, client, domain="aps", prompts=custom_prompts)

        with patch("scout_ai.providers.pageindex.indexer._default_prompt") as mock_dp:
            result = indexer._get_prompt("TOC_DETECT_PROMPT")
            assert result == "custom injected prompt"
            mock_dp.assert_not_called()
