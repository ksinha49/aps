"""Tests for the feedback capture API endpoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from scout_ai.api.auth import require_auth
from scout_ai.api.routes import feedback
from scout_ai.core.config import AppSettings
from scout_ai.persistence.memory_backend import MemoryPersistenceBackend


def _build_feedback_app() -> tuple[FastAPI, MemoryPersistenceBackend]:
    """Build a minimal FastAPI app with the feedback router and in-memory persistence."""
    settings = AppSettings()
    backend = MemoryPersistenceBackend()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.settings = settings
        app.state.persistence_backend = backend
        yield

    app = FastAPI(lifespan=lifespan)
    app.include_router(
        feedback.router,
        prefix="/api",
        dependencies=[Depends(require_auth)],
    )
    return app, backend


def _valid_payload() -> dict[str, object]:
    """Return a minimal valid feedback payload."""
    return {
        "doc_id": "doc-123",
        "question_id": "q-001",
        "corrected_answer": "The corrected answer text.",
    }


class TestFeedbackEndpointAcceptsCorrection:
    """POST /api/feedback with valid data returns 201 with status='recorded'."""

    def test_feedback_endpoint_accepts_correction(self) -> None:
        app, _ = _build_feedback_app()
        with TestClient(app) as client:
            resp = client.post("/api/feedback", json=_valid_payload())
            assert resp.status_code == 201
            body = resp.json()
            assert body["status"] == "recorded"


class TestFeedbackEndpointReturnsFeedbackId:
    """POST /api/feedback returns a non-empty feedback_id."""

    def test_feedback_endpoint_returns_feedback_id(self) -> None:
        app, _ = _build_feedback_app()
        with TestClient(app) as client:
            resp = client.post("/api/feedback", json=_valid_payload())
            assert resp.status_code == 201
            body = resp.json()
            assert "feedback_id" in body
            assert isinstance(body["feedback_id"], str)
            assert len(body["feedback_id"]) > 0
            # Verify the feedback_id contains the doc_id and question_id
            assert "doc-123" in body["feedback_id"]
            assert "q-001" in body["feedback_id"]


class TestFeedbackEndpointValidatesRequiredFields:
    """POST /api/feedback with missing required fields returns 422."""

    def test_missing_doc_id(self) -> None:
        app, _ = _build_feedback_app()
        payload = {
            "question_id": "q-001",
            "corrected_answer": "Some answer.",
        }
        with TestClient(app) as client:
            resp = client.post("/api/feedback", json=payload)
            assert resp.status_code == 422

    def test_missing_question_id(self) -> None:
        app, _ = _build_feedback_app()
        payload = {
            "doc_id": "doc-123",
            "corrected_answer": "Some answer.",
        }
        with TestClient(app) as client:
            resp = client.post("/api/feedback", json=payload)
            assert resp.status_code == 422

    def test_missing_corrected_answer(self) -> None:
        app, _ = _build_feedback_app()
        payload = {
            "doc_id": "doc-123",
            "question_id": "q-001",
        }
        with TestClient(app) as client:
            resp = client.post("/api/feedback", json=payload)
            assert resp.status_code == 422

    def test_empty_body(self) -> None:
        app, _ = _build_feedback_app()
        with TestClient(app) as client:
            resp = client.post("/api/feedback", json={})
            assert resp.status_code == 422


class TestFeedbackResponseIncludesRecordedStatus:
    """The response status field is always 'recorded'."""

    def test_feedback_response_includes_recorded_status(self) -> None:
        app, _ = _build_feedback_app()
        with TestClient(app) as client:
            resp = client.post("/api/feedback", json=_valid_payload())
            assert resp.status_code == 201
            assert resp.json()["status"] == "recorded"

    def test_full_payload_returns_recorded_status(self) -> None:
        app, _ = _build_feedback_app()
        payload = {
            "doc_id": "doc-456",
            "question_id": "q-002",
            "corrected_answer": "Updated answer.",
            "corrected_citations": [{"page": 5, "text": "source passage"}],
            "auditor_id": "auditor-jane",
            "notes": "Original answer missed key detail.",
        }
        with TestClient(app) as client:
            resp = client.post("/api/feedback", json=payload)
            assert resp.status_code == 201
            body = resp.json()
            assert body["status"] == "recorded"
            assert "doc-456" in body["feedback_id"]


class TestFeedbackPersistence:
    """Verify that feedback is persisted via the backend."""

    def test_payload_is_saved_to_persistence_backend(self) -> None:
        app, backend = _build_feedback_app()
        with TestClient(app) as client:
            resp = client.post("/api/feedback", json=_valid_payload())
            assert resp.status_code == 201
            feedback_id = resp.json()["feedback_id"]

        # The backend should have exactly one key under the feedback/ prefix
        keys = backend.list_keys(prefix="feedback/")
        assert len(keys) == 1
        assert feedback_id in keys[0]
