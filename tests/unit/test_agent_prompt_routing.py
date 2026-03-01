"""Tests for domain-routed agent system prompts."""

from __future__ import annotations

from unittest.mock import patch


class TestIndexingAgentPromptRouting:
    """Prompt routing for the indexing agent factory."""

    def test_resolves_domain_prompt(self) -> None:
        """Should use the domain-specific prompt when available."""
        with patch("scout_ai.agents.indexing_agent.get_prompt") as mock_get:
            mock_get.return_value = "Custom domain prompt"
            from scout_ai.agents.indexing_agent import _resolve_system_prompt

            result = _resolve_system_prompt("workers_comp")
            mock_get.assert_called_once_with("workers_comp", "indexing_agent", "INDEXING_SYSTEM_PROMPT")
            assert result == "Custom domain prompt"

    def test_falls_back_to_base(self) -> None:
        """Should fall back to base when domain prompt is not found."""
        with patch("scout_ai.agents.indexing_agent.get_prompt") as mock_get:
            mock_get.side_effect = [KeyError("not found"), "Base prompt"]
            from scout_ai.agents.indexing_agent import _resolve_system_prompt

            result = _resolve_system_prompt("unknown_domain")
            assert result == "Base prompt"
            assert mock_get.call_count == 2
            mock_get.assert_any_call("unknown_domain", "indexing_agent", "INDEXING_SYSTEM_PROMPT")
            mock_get.assert_any_call("base", "indexing_agent", "INDEXING_SYSTEM_PROMPT")

    def test_base_domain_resolves_directly(self) -> None:
        """When domain is 'base', first call already hits the base module."""
        with patch("scout_ai.agents.indexing_agent.get_prompt") as mock_get:
            mock_get.return_value = "Base indexing prompt"
            from scout_ai.agents.indexing_agent import _resolve_system_prompt

            result = _resolve_system_prompt("base")
            mock_get.assert_called_once_with("base", "indexing_agent", "INDEXING_SYSTEM_PROMPT")
            assert result == "Base indexing prompt"


class TestRetrievalAgentPromptRouting:
    """Prompt routing for the retrieval agent factory."""

    def test_resolves_domain_prompt(self) -> None:
        """Should use the domain-specific prompt when available."""
        with patch("scout_ai.agents.retrieval_agent.get_prompt") as mock_get:
            mock_get.return_value = "Custom retrieval prompt"
            from scout_ai.agents.retrieval_agent import _resolve_system_prompt

            result = _resolve_system_prompt("workers_comp")
            mock_get.assert_called_once_with("workers_comp", "retrieval_agent", "RETRIEVAL_SYSTEM_PROMPT")
            assert result == "Custom retrieval prompt"

    def test_falls_back_to_base(self) -> None:
        """Should fall back to base when domain prompt is not found."""
        with patch("scout_ai.agents.retrieval_agent.get_prompt") as mock_get:
            mock_get.side_effect = [KeyError("not found"), "Base retrieval prompt"]
            from scout_ai.agents.retrieval_agent import _resolve_system_prompt

            result = _resolve_system_prompt("unknown_domain")
            assert result == "Base retrieval prompt"
            assert mock_get.call_count == 2
            mock_get.assert_any_call("unknown_domain", "retrieval_agent", "RETRIEVAL_SYSTEM_PROMPT")
            mock_get.assert_any_call("base", "retrieval_agent", "RETRIEVAL_SYSTEM_PROMPT")

    def test_base_domain_resolves_directly(self) -> None:
        """When domain is 'base', first call already hits the base module."""
        with patch("scout_ai.agents.retrieval_agent.get_prompt") as mock_get:
            mock_get.return_value = "Base retrieval prompt"
            from scout_ai.agents.retrieval_agent import _resolve_system_prompt

            result = _resolve_system_prompt("base")
            mock_get.assert_called_once_with("base", "retrieval_agent", "RETRIEVAL_SYSTEM_PROMPT")
            assert result == "Base retrieval prompt"


class TestExtractionAgentPromptRouting:
    """Prompt routing for the extraction agent factory."""

    def test_resolves_domain_prompt(self) -> None:
        """Should use the domain-specific prompt when available."""
        with patch("scout_ai.agents.extraction_agent.get_prompt") as mock_get:
            mock_get.return_value = "Custom extraction prompt"
            from scout_ai.agents.extraction_agent import _resolve_system_prompt

            result = _resolve_system_prompt("workers_comp")
            mock_get.assert_called_once_with("workers_comp", "extraction_agent", "EXTRACTION_SYSTEM_PROMPT")
            assert result == "Custom extraction prompt"

    def test_falls_back_to_base(self) -> None:
        """Should fall back to base when domain prompt is not found."""
        with patch("scout_ai.agents.extraction_agent.get_prompt") as mock_get:
            mock_get.side_effect = [KeyError("not found"), "Base extraction prompt"]
            from scout_ai.agents.extraction_agent import _resolve_system_prompt

            result = _resolve_system_prompt("unknown_domain")
            assert result == "Base extraction prompt"
            assert mock_get.call_count == 2
            mock_get.assert_any_call("unknown_domain", "extraction_agent", "EXTRACTION_SYSTEM_PROMPT")
            mock_get.assert_any_call("base", "extraction_agent", "EXTRACTION_SYSTEM_PROMPT")

    def test_base_domain_resolves_directly(self) -> None:
        """When domain is 'base', first call already hits the base module."""
        with patch("scout_ai.agents.extraction_agent.get_prompt") as mock_get:
            mock_get.return_value = "Base extraction prompt"
            from scout_ai.agents.extraction_agent import _resolve_system_prompt

            result = _resolve_system_prompt("base")
            mock_get.assert_called_once_with("base", "extraction_agent", "EXTRACTION_SYSTEM_PROMPT")
            assert result == "Base extraction prompt"
