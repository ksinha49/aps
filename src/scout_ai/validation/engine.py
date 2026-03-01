"""Re-export shim — canonical location is ``scout_ai.domains.aps.validation.engine``.

The ``RulesEngine`` has moved to ``domains.aps.validation.engine``.
This module re-exports it for backward compatibility::

    from scout_ai.validation.engine import RulesEngine  # still works
"""

from __future__ import annotations

from scout_ai.domains.aps.validation.engine import RulesEngine

__all__ = ["RulesEngine"]
