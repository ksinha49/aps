"""Feedback capture endpoint for human-in-the-loop corrections."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    """Payload for submitting a correction to an extraction result."""

    doc_id: str
    question_id: str
    corrected_answer: str
    corrected_citations: list[dict[str, object]] = Field(default_factory=list)
    auditor_id: str = ""
    notes: str = ""


class FeedbackResponse(BaseModel):
    """Acknowledgement returned after feedback is recorded."""

    feedback_id: str
    status: str = "recorded"


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(request: FeedbackRequest, req: Request) -> FeedbackResponse:
    """Record a human correction for an extraction result.

    Persists the feedback payload via the configured persistence backend
    under the ``feedback/{feedback_id}`` key.
    """
    settings = req.app.state.settings
    recorded_at = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    feedback_id = f"{request.doc_id}_{request.question_id}_{timestamp}"

    payload = {
        "feedback_id": feedback_id,
        "doc_id": request.doc_id,
        "question_id": request.question_id,
        "corrected_answer": request.corrected_answer,
        "corrected_citations": request.corrected_citations,
        "auditor_id": request.auditor_id,
        "notes": request.notes,
        "recorded_at": recorded_at,
    }

    # Persist via the persistence backend matching existing patterns
    from scout_ai.persistence import FilePersistenceBackend, MemoryPersistenceBackend
    from scout_ai.persistence.protocols import IPersistenceBackend

    backend: IPersistenceBackend
    if hasattr(req.app.state, "persistence_backend"):
        backend = req.app.state.persistence_backend
    elif settings.persistence.backend == "memory":
        backend = MemoryPersistenceBackend()
    else:
        backend = FilePersistenceBackend(base_path=settings.persistence.store_path)

    key = f"feedback/{feedback_id}"
    backend.save(key, json.dumps(payload))
    log.info("Feedback recorded", extra={"feedback_id": feedback_id, "key": key})

    return FeedbackResponse(feedback_id=feedback_id)
