"""Re-export shim — canonical location is ``scout_ai.domains.aps.synthesis.pipeline``.

The ``SynthesisPipeline`` has moved to ``domains.aps.synthesis.pipeline``.
This module re-exports it for backward compatibility::

    from scout_ai.synthesis.pipeline import SynthesisPipeline  # still works
"""

from __future__ import annotations

from scout_ai.domains.aps.synthesis.pipeline import SynthesisPipeline

__all__ = ["SynthesisPipeline"]
