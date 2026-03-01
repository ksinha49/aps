"""Pluggable inference backend layer.

Usage::

    from scout_ai.inference import (
        IInferenceBackend,
        InferenceRequest,
        InferenceResult,
        RealTimeBackend,
        create_inference_backend,
    )
"""

from __future__ import annotations

from scout_ai.inference.factory import create_inference_backend
from scout_ai.inference.protocols import IInferenceBackend, InferenceRequest, InferenceResult
from scout_ai.inference.realtime import RealTimeBackend

__all__ = [
    "IInferenceBackend",
    "InferenceRequest",
    "InferenceResult",
    "RealTimeBackend",
    "create_inference_backend",
]
