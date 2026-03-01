"""Tests for OTLP tracing configuration."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scout_ai.core.config import ObservabilityConfig


class TestObservabilityConfig:
    """Verify default config values."""

    def test_otlp_insecure_defaults_to_true(self) -> None:
        config = ObservabilityConfig()
        assert config.otlp_insecure is True

    def test_otlp_insecure_can_be_disabled(self) -> None:
        config = ObservabilityConfig(otlp_insecure=False)
        assert config.otlp_insecure is False


def _mock_otel_modules() -> dict[str, Any]:
    """Build a sys.modules dict that stubs out all OpenTelemetry packages."""
    mock_trace = MagicMock()
    mock_resource_cls = MagicMock()
    mock_provider_cls = MagicMock()
    mock_exporter_cls = MagicMock()
    mock_processor_cls = MagicMock()
    return {
        "opentelemetry": MagicMock(trace=mock_trace),
        "opentelemetry.trace": mock_trace,
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": MagicMock(
            OTLPSpanExporter=mock_exporter_cls,
        ),
        "opentelemetry.sdk.resources": MagicMock(Resource=mock_resource_cls),
        "opentelemetry.sdk.trace": MagicMock(TracerProvider=mock_provider_cls),
        "opentelemetry.sdk.trace.export": MagicMock(BatchSpanProcessor=mock_processor_cls),
        # Expose mocks for assertions
        "_resource_cls": mock_resource_cls,
        "_exporter_cls": mock_exporter_cls,
    }


class TestSetupTracing:
    """Verify setup_tracing behavior."""

    def test_noop_when_disabled(self) -> None:
        """No errors when tracing is disabled, even without otel installed."""
        from scout_ai.hooks.tracing import setup_tracing

        config = ObservabilityConfig(enable_tracing=False)
        setup_tracing(config)  # Should not raise

    def test_warns_on_missing_otel(self) -> None:
        """Graceful warning when otel packages not installed."""
        config = ObservabilityConfig(enable_tracing=True)

        # Setting sys.modules entries to None causes ImportError on import
        null_modules = {
            "opentelemetry": None,
            "opentelemetry.trace": None,
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": None,
            "opentelemetry.sdk.resources": None,
            "opentelemetry.sdk.trace": None,
            "opentelemetry.sdk.trace.export": None,
        }

        with patch.dict("sys.modules", null_modules):
            import scout_ai.hooks.tracing as tracing_mod

            importlib.reload(tracing_mod)

            with patch.object(tracing_mod, "log") as mock_log:
                tracing_mod.setup_tracing(config)

            mock_log.warning.assert_called_once()

    def test_resource_attrs_include_tenant(self) -> None:
        """When tenant_id and lob are provided, custom attributes are set."""
        config = ObservabilityConfig(enable_tracing=True)
        modules = _mock_otel_modules()

        with patch.dict("sys.modules", modules):
            import scout_ai.hooks.tracing as tracing_mod

            importlib.reload(tracing_mod)
            tracing_mod.setup_tracing(config, tenant_id="ameritas-rp", lob="group-health")

        mock_resource_cls = modules["_resource_cls"]
        mock_resource_cls.create.assert_called_once()
        attrs = mock_resource_cls.create.call_args[0][0]
        assert attrs["service.name"] == "scout-ai"
        assert attrs["scout.tenant_id"] == "ameritas-rp"
        assert attrs["scout.lob"] == "group-health"

    def test_resource_attrs_omit_empty_tenant(self) -> None:
        """When tenant_id and lob are empty, custom attributes are NOT set."""
        config = ObservabilityConfig(enable_tracing=True)
        modules = _mock_otel_modules()

        with patch.dict("sys.modules", modules):
            import scout_ai.hooks.tracing as tracing_mod

            importlib.reload(tracing_mod)
            tracing_mod.setup_tracing(config, tenant_id="", lob="")

        mock_resource_cls = modules["_resource_cls"]
        attrs = mock_resource_cls.create.call_args[0][0]
        assert attrs == {"service.name": "scout-ai"}
        assert "scout.tenant_id" not in attrs
        assert "scout.lob" not in attrs

    @pytest.mark.parametrize("insecure", [True, False])
    def test_insecure_flag_passed_to_exporter(self, insecure: bool) -> None:
        """The insecure flag from config is passed to OTLPSpanExporter."""
        config = ObservabilityConfig(enable_tracing=True, otlp_insecure=insecure)
        modules = _mock_otel_modules()

        with patch.dict("sys.modules", modules):
            import scout_ai.hooks.tracing as tracing_mod

            importlib.reload(tracing_mod)
            tracing_mod.setup_tracing(config)

        mock_exporter_cls = modules["_exporter_cls"]
        mock_exporter_cls.assert_called_once_with(
            endpoint="http://localhost:4317",
            insecure=insecure,
        )
