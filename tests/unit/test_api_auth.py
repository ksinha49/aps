"""Tests for API authentication (JWT + API key)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import pytest
from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient

from scout_ai.api.auth import require_auth
from scout_ai.api.routes import health
from scout_ai.core.config import AppSettings, AuthConfig


def _build_app_with_auth(settings: AppSettings) -> FastAPI:
    """Build a FastAPI app with auth dependency on /api routes, mirroring production."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.settings = settings
        yield

    app = FastAPI(lifespan=lifespan)
    app.include_router(health.router)

    protected_router = APIRouter(prefix="/api", dependencies=[Depends(require_auth)])

    @protected_router.get("/ping")
    async def ping() -> dict[str, str]:
        return {"pong": "ok"}

    app.include_router(protected_router)
    return app


def _make_settings(*, auth_enabled: bool = False, api_keys: list[str] | None = None) -> AppSettings:
    """Create AppSettings with the given auth configuration."""
    settings = AppSettings()
    settings.auth = AuthConfig(enabled=auth_enabled, api_keys=api_keys or [])
    return settings


class TestHealthAlwaysAccessible:
    """Health endpoints should never require authentication."""

    def test_health_no_auth(self) -> None:
        app = _build_app_with_auth(_make_settings(auth_enabled=True, api_keys=["secret"]))
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_ready_no_auth(self) -> None:
        app = _build_app_with_auth(_make_settings(auth_enabled=True, api_keys=["secret"]))
        with TestClient(app) as client:
            resp = client.get("/ready")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ready"


class TestAuthEnabled401:
    """When auth is enabled and no credentials are provided, /api/* returns 401."""

    def test_no_credentials_returns_401(self) -> None:
        app = _build_app_with_auth(_make_settings(auth_enabled=True, api_keys=["secret"]))
        with TestClient(app) as client:
            resp = client.get("/api/ping")
            assert resp.status_code == 401
            assert "Authentication required" in resp.json()["detail"]

    def test_wrong_api_key_returns_401(self) -> None:
        app = _build_app_with_auth(_make_settings(auth_enabled=True, api_keys=["correct-key"]))
        with TestClient(app) as client:
            resp = client.get("/api/ping", headers={"X-API-Key": "wrong-key"})
            assert resp.status_code == 401

    def test_empty_api_key_returns_401(self) -> None:
        app = _build_app_with_auth(_make_settings(auth_enabled=True, api_keys=["secret"]))
        with TestClient(app) as client:
            resp = client.get("/api/ping", headers={"X-API-Key": ""})
            assert resp.status_code == 401


class TestValidApiKeyAccess:
    """A valid API key should grant access to protected routes."""

    def test_valid_api_key_allows_access(self) -> None:
        app = _build_app_with_auth(_make_settings(auth_enabled=True, api_keys=["my-secret-key"]))
        with TestClient(app) as client:
            resp = client.get("/api/ping", headers={"X-API-Key": "my-secret-key"})
            assert resp.status_code == 200
            assert resp.json() == {"pong": "ok"}

    def test_multiple_api_keys(self) -> None:
        app = _build_app_with_auth(_make_settings(auth_enabled=True, api_keys=["key-1", "key-2", "key-3"]))
        with TestClient(app) as client:
            for key in ["key-1", "key-2", "key-3"]:
                resp = client.get("/api/ping", headers={"X-API-Key": key})
                assert resp.status_code == 200


class TestAuthDisabled:
    """When auth is disabled, all requests should pass through without credentials."""

    def test_disabled_allows_api_access(self) -> None:
        app = _build_app_with_auth(_make_settings(auth_enabled=False))
        with TestClient(app) as client:
            resp = client.get("/api/ping")
            assert resp.status_code == 200
            assert resp.json() == {"pong": "ok"}

    def test_disabled_ignores_api_key(self) -> None:
        app = _build_app_with_auth(_make_settings(auth_enabled=False))
        with TestClient(app) as client:
            resp = client.get("/api/ping", headers={"X-API-Key": "anything"})
            assert resp.status_code == 200

    def test_disabled_health_still_works(self) -> None:
        app = _build_app_with_auth(_make_settings(auth_enabled=False))
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200


class TestAuthConfigDefaults:
    """AuthConfig should have sensible defaults and support env var overrides."""

    def test_defaults(self) -> None:
        cfg = AuthConfig()
        assert cfg.enabled is False
        assert cfg.jwks_url == ""
        assert cfg.issuer == ""
        assert cfg.audience == "scout-ai"
        assert cfg.algorithm == "RS256"
        assert cfg.tenant_claim == "custom:tenant_id"
        assert cfg.api_key_header == "X-API-Key"
        assert cfg.api_keys == []

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCOUT_AUTH_ENABLED", "true")
        monkeypatch.setenv("SCOUT_AUTH_AUDIENCE", "my-app")
        monkeypatch.setenv("SCOUT_AUTH_ALGORITHM", "HS256")

        cfg = AuthConfig()
        assert cfg.enabled is True
        assert cfg.audience == "my-app"
        assert cfg.algorithm == "HS256"

    def test_app_settings_includes_auth(self) -> None:
        settings = AppSettings()
        assert hasattr(settings, "auth")
        assert isinstance(settings.auth, AuthConfig)
        assert settings.auth.enabled is False
