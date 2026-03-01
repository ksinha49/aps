"""Inference backend factory — resolves backend from config."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scout_ai.inference.protocols import IInferenceBackend
from scout_ai.inference.realtime import RealTimeBackend

if TYPE_CHECKING:
    from scout_ai.core.config import AppSettings

log = logging.getLogger(__name__)


def create_inference_backend(settings: AppSettings) -> IInferenceBackend:
    """Create an inference backend based on settings.

    When ``settings.llm.inference_backend`` is ``"realtime"``, returns
    the built-in :class:`RealTimeBackend`.

    When it's a dotted path like ``mypackage.backends:BedrockBatchBackend``,
    imports and instantiates the external class, passing ``settings`` to
    the constructor.

    Args:
        settings: Application settings.

    Returns:
        An inference backend instance.

    Raises:
        ImportError: If the dotted-path class cannot be found.
        TypeError: If the resolved object is not callable.
    """
    backend_spec = settings.llm.inference_backend

    if backend_spec == "realtime":
        log.info("Using built-in RealTimeBackend")
        return RealTimeBackend(max_concurrent=settings.retrieval.max_concurrent)

    # External dotted-path: reuse the same import helper as DomainConfig
    from scout_ai.domains.registry import _import_dotted_path

    log.info("Loading external inference backend: %s", backend_spec)
    cls = _import_dotted_path(backend_spec)

    if not callable(cls):
        raise TypeError(
            f"Inference backend {backend_spec!r} resolved to {cls!r}, "
            "which is not callable"
        )

    return cls(settings)
