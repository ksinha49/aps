"""Per-run analytics tracker using ContextVars.

Follows the same ContextVar pattern as ``cost_hook.py`` — opt-in, backward
compatible, and zero overhead when no run is active.

Usage::

    analytics = start_run(doc_id="my-doc")
    with track_stage("retrieval") as stage:
        stage.success_count = 5
    analytics = end_run()
    print(analytics.total_duration_ms)
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Generator

from scout_ai.models import RunAnalytics, StageMetrics

_current_run: ContextVar[RunAnalytics | None] = ContextVar("scout_current_run", default=None)


def get_current_run() -> RunAnalytics | None:
    """Get the active RunAnalytics, or None if no run is active."""
    return _current_run.get()


def start_run(doc_id: str = "", run_id: str | None = None) -> RunAnalytics:
    """Create and activate a new RunAnalytics for the current context."""
    analytics = RunAnalytics(
        run_id=run_id or uuid.uuid4().hex[:12],
        doc_id=doc_id,
        started_at=datetime.now(timezone.utc),
        status="running",
    )
    _current_run.set(analytics)

    # Bind run_id to structlog context if available
    try:
        import structlog

        structlog.contextvars.bind_contextvars(run_id=analytics.run_id)
    except ImportError:
        pass

    return analytics


def end_run() -> RunAnalytics | None:
    """Finalize the current run and return its analytics. Returns None if no run is active."""
    analytics = _current_run.get()
    if analytics is None:
        return None

    analytics.finalize()
    _current_run.set(None)

    # Unbind run_id from structlog context
    try:
        import structlog

        structlog.contextvars.unbind_contextvars("run_id")
    except ImportError:
        pass

    return analytics


@contextmanager
def track_stage(name: str) -> Generator[StageMetrics, None, None]:
    """Context manager that records a StageMetrics entry on the current run.

    No-op if no run is active (opt-in, backward compatible).
    """
    analytics = _current_run.get()

    stage = StageMetrics(stage=name, started_at=datetime.now(timezone.utc))

    # Bind stage name to structlog context
    try:
        import structlog

        structlog.contextvars.bind_contextvars(stage=name)
    except ImportError:
        pass

    try:
        yield stage
    finally:
        stage.ended_at = datetime.now(timezone.utc)
        if stage.started_at:
            stage.duration_ms = (stage.ended_at - stage.started_at).total_seconds() * 1000

        if analytics is not None:
            analytics.stages.append(stage)

        # Unbind stage from structlog context
        try:
            import structlog

            structlog.contextvars.unbind_contextvars("stage")
        except ImportError:
            pass
