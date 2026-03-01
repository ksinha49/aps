"""Tests for the async indexing endpoint (SQS-backed)."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from unittest.mock import patch

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from scout_ai.api.auth import require_auth
from scout_ai.api.routes import index_async
from scout_ai.core.config import AppSettings


def _build_app(settings: AppSettings) -> FastAPI:
    """Build a minimal FastAPI app with the async index route."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.settings = settings
        yield

    app = FastAPI(lifespan=lifespan)
    app.include_router(
        index_async.router, prefix="/api", dependencies=[Depends(require_auth)]
    )
    return app


_VALID_PAYLOAD = {
    "doc_id": "doc-001",
    "doc_name": "Test Document",
    "pages": [{"page_number": 1, "text": "Hello world"}],
}


class TestAsyncIndexReturnsJobId:
    """POST /api/index/async returns 202 with a job_id when SQS is configured."""

    def test_async_index_returns_job_id(self) -> None:
        settings = AppSettings()
        settings.index_queue_url = "https://sqs.us-east-2.amazonaws.com/123456789012/scout-index"
        app = _build_app(settings)

        fake_job_id = str(uuid.uuid4())

        with patch.object(index_async, "_send_to_queue", return_value=fake_job_id) as mock_send:
            with TestClient(app) as client:
                resp = client.post("/api/index/async", json=_VALID_PAYLOAD)

            assert resp.status_code == 202
            body = resp.json()
            assert body["job_id"] == fake_job_id
            assert body["status"] == "queued"
            assert "queued" in body["message"].lower()

            mock_send.assert_called_once()
            call_kwargs = mock_send.call_args
            assert call_kwargs[1]["queue_url"] == settings.index_queue_url


class TestAsyncIndexReturns501WhenNotConfigured:
    """POST /api/index/async returns 501 when index_queue_url is empty."""

    def test_async_index_returns_501_when_not_configured(self) -> None:
        settings = AppSettings()
        # index_queue_url defaults to ""
        app = _build_app(settings)

        with TestClient(app) as client:
            resp = client.post("/api/index/async", json=_VALID_PAYLOAD)

        assert resp.status_code == 501
        assert "SCOUT_INDEX_QUEUE_URL" in resp.json()["detail"]


class TestAsyncIndexRequestValidation:
    """POST /api/index/async returns 422 for invalid request bodies."""

    def test_missing_doc_id(self) -> None:
        settings = AppSettings()
        settings.index_queue_url = "https://sqs.us-east-2.amazonaws.com/123/q"
        app = _build_app(settings)

        with TestClient(app) as client:
            resp = client.post(
                "/api/index/async",
                json={"doc_name": "Test", "pages": [{"page_number": 1, "text": "x"}]},
            )

        assert resp.status_code == 422

    def test_missing_pages(self) -> None:
        settings = AppSettings()
        settings.index_queue_url = "https://sqs.us-east-2.amazonaws.com/123/q"
        app = _build_app(settings)

        with TestClient(app) as client:
            resp = client.post(
                "/api/index/async",
                json={"doc_id": "d1", "doc_name": "Test"},
            )

        assert resp.status_code == 422

    def test_missing_all_fields(self) -> None:
        settings = AppSettings()
        settings.index_queue_url = "https://sqs.us-east-2.amazonaws.com/123/q"
        app = _build_app(settings)

        with TestClient(app) as client:
            resp = client.post("/api/index/async", json={})

        assert resp.status_code == 422


class TestAsyncIndexJobIdIsUuid:
    """Verify job_id returned is a valid UUID."""

    def test_async_index_job_id_is_uuid(self) -> None:
        settings = AppSettings()
        settings.index_queue_url = "https://sqs.us-east-2.amazonaws.com/123/q"
        app = _build_app(settings)

        real_uuid = str(uuid.uuid4())

        with patch.object(index_async, "_send_to_queue", return_value=real_uuid):
            with TestClient(app) as client:
                resp = client.post("/api/index/async", json=_VALID_PAYLOAD)

        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        # Verify it parses as a valid UUID (raises ValueError if not)
        parsed = uuid.UUID(job_id)
        assert str(parsed) == job_id
