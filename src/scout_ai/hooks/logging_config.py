"""Structured logging configuration using structlog.

Provides JSON logs in production and colored console output in development.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scout_ai.core.config import ObservabilityConfig


def setup_logging(config: ObservabilityConfig) -> None:
    """Configure structured logging.

    Falls back to stdlib logging if structlog is not installed.
    """
    level = getattr(logging, config.log_level.upper(), logging.INFO)

    try:
        import structlog

        shared_processors: list = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
        ]

        if sys.stderr.isatty():
            # Dev mode: colored console
            renderer = structlog.dev.ConsoleRenderer()
        else:
            # Production: JSON lines
            renderer = structlog.processors.JSONRenderer()

        structlog.configure(
            processors=[
                *shared_processors,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)

        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level)

    except ImportError:
        # structlog not installed — use stdlib
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
            stream=sys.stderr,
        )

    logging.getLogger("scout_ai").setLevel(level)
