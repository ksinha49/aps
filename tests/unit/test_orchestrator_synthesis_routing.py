"""Tests for domain-registry-routed synthesis in ExtractionPipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scout_ai.agents.orchestrator import ExtractionPipeline

_REGISTRY_PATH = "scout_ai.domains.registry.get_registry"
_IMPORT_PATH = "scout_ai.domains.registry._import_dotted_path"


class TestSynthesisRouting:
    """Verify that run_with_synthesis resolves synthesis via the domain registry."""

    def test_domain_stored_on_pipeline(self) -> None:
        pipeline = ExtractionPipeline(
            retrieval_provider=AsyncMock(),
            chat_provider=AsyncMock(),
            domain="workers_comp",
        )
        assert pipeline._domain == "workers_comp"

    def test_default_domain_is_aps(self) -> None:
        pipeline = ExtractionPipeline(
            retrieval_provider=AsyncMock(),
            chat_provider=AsyncMock(),
        )
        assert pipeline._domain == "aps"

    @pytest.mark.asyncio
    async def test_unknown_domain_returns_none_summary(self) -> None:
        """An unregistered domain should log a warning and return (results, None, None)."""
        pipeline = ExtractionPipeline(
            retrieval_provider=AsyncMock(),
            chat_provider=AsyncMock(),
            domain="nonexistent_domain",
        )
        pipeline.run = AsyncMock(return_value=[])

        from scout_ai.domains.registry import DomainRegistry

        empty_reg = DomainRegistry()
        with patch(_REGISTRY_PATH, return_value=empty_reg):
            results, summary, report = await pipeline.run_with_synthesis(
                index=MagicMock(),
                questions=[],
                synthesize=True,
            )
        assert results == []
        assert summary is None
        assert report is None

    @pytest.mark.asyncio
    async def test_domain_without_synthesis_pipeline_returns_none(self) -> None:
        """A registered domain with no synthesis_pipeline dotted-path should return None."""
        from scout_ai.domains.registry import DomainConfig, DomainRegistry

        pipeline = ExtractionPipeline(
            retrieval_provider=AsyncMock(),
            chat_provider=AsyncMock(),
            domain="test_domain",
        )
        pipeline.run = AsyncMock(return_value=[])

        reg = DomainRegistry()
        reg.register(DomainConfig(name="test_domain", display_name="Test"))
        with patch(_REGISTRY_PATH, return_value=reg):
            results, summary, report = await pipeline.run_with_synthesis(
                index=MagicMock(),
                questions=[],
                synthesize=True,
            )
        assert results == []
        assert summary is None
        assert report is None

    @pytest.mark.asyncio
    async def test_domain_synthesis_pipeline_resolved_from_registry(self) -> None:
        """When the domain registry has a synthesis_pipeline, it should be resolved and used."""
        from scout_ai.domains.registry import DomainConfig, DomainRegistry

        mock_synth_instance = MagicMock()
        mock_synth_instance.synthesize = AsyncMock(return_value={"narrative": "ok"})
        MockSynthCls = MagicMock(return_value=mock_synth_instance)

        chat_provider = AsyncMock()
        chat_provider._client = MagicMock()
        chat_provider._cache_enabled = False

        pipeline = ExtractionPipeline(
            retrieval_provider=AsyncMock(),
            chat_provider=chat_provider,
            domain="aps_test",
        )
        pipeline.run = AsyncMock(return_value=[])

        reg = DomainRegistry()
        reg.register(
            DomainConfig(
                name="aps_test",
                display_name="APS Test",
                synthesis_pipeline="scout_ai.domains.aps.synthesis.pipeline:SynthesisPipeline",
            )
        )

        with (
            patch(_REGISTRY_PATH, return_value=reg),
            patch(_IMPORT_PATH, return_value=MockSynthCls),
        ):
            results, summary, report = await pipeline.run_with_synthesis(
                index=MagicMock(doc_id="d1", doc_name="test"),
                questions=[],
                synthesize=True,
            )

        MockSynthCls.assert_called_once_with(chat_provider._client, cache_enabled=False)
        mock_synth_instance.synthesize.assert_awaited_once()
        assert summary == {"narrative": "ok"}

    @pytest.mark.asyncio
    async def test_no_client_attribute_returns_none(self) -> None:
        """When the chat provider has no _client, synthesis should bail out."""
        from scout_ai.domains.registry import DomainConfig, DomainRegistry

        MockSynthCls = MagicMock()

        # Chat provider without _client attribute
        chat_provider = MagicMock(spec=[])

        pipeline = ExtractionPipeline(
            retrieval_provider=AsyncMock(),
            chat_provider=chat_provider,
            domain="aps_test",
        )
        pipeline.run = AsyncMock(return_value=[])

        reg = DomainRegistry()
        reg.register(
            DomainConfig(
                name="aps_test",
                display_name="APS Test",
                synthesis_pipeline="scout_ai.domains.aps.synthesis.pipeline:SynthesisPipeline",
            )
        )

        with (
            patch(_REGISTRY_PATH, return_value=reg),
            patch(_IMPORT_PATH, return_value=MockSynthCls),
        ):
            results, summary, report = await pipeline.run_with_synthesis(
                index=MagicMock(doc_id="d1", doc_name="test"),
                questions=[],
                synthesize=True,
            )

        MockSynthCls.assert_not_called()
        assert summary is None

    @pytest.mark.asyncio
    async def test_synthesize_false_skips_resolution(self) -> None:
        """When synthesize=False, no registry lookup should occur."""
        pipeline = ExtractionPipeline(
            retrieval_provider=AsyncMock(),
            chat_provider=AsyncMock(),
            domain="nonexistent_domain",
        )
        pipeline.run = AsyncMock(return_value=[])

        results, summary, report = await pipeline.run_with_synthesis(
            index=MagicMock(),
            questions=[],
            synthesize=False,
        )
        assert results == []
        assert summary is None
        assert report is None
