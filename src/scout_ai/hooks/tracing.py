"""OpenTelemetry tracing setup for Strands agents.

Configures OTLP exporter when ``settings.observability.enable_tracing`` is True.
Traces flow: Agent → Cycle → LLM Call → Tool Call (Strands built-in hierarchy).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scout_ai.core.config import ObservabilityConfig

log = logging.getLogger(__name__)


def setup_tracing(config: ObservabilityConfig) -> None:
    """Initialize OpenTelemetry tracing if enabled in config.

    Requires ``strands-agents[otel]`` to be installed.  Does nothing
    if tracing is disabled.
    """
    if not config.enable_tracing:
        log.debug("Tracing disabled — skipping OpenTelemetry setup")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning(
            "OpenTelemetry packages not found. "
            "Install with: pip install scout-ai[otel]"
        )
        return

    resource = Resource.create({
        "service.name": config.service_name,
    })

    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    log.info(
        "OpenTelemetry tracing enabled — exporting to %s as '%s'",
        config.otlp_endpoint,
        config.service_name,
    )
