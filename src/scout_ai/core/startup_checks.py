"""Startup validation: fail-fast on critical misconfigurations."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scout_ai.core.config import AppSettings

log = logging.getLogger(__name__)

# Providers that use IAM/local auth and do not require an API key
_NO_KEY_PROVIDERS = frozenset({"bedrock", "ollama"})


def validate_settings(settings: AppSettings) -> None:
    """Validate application settings at startup. Raises ValueError on fatal misconfig."""
    _check_api_key(settings)
    _check_persistence(settings)
    _check_auth(settings)


def _check_api_key(settings: AppSettings) -> None:
    """Reject placeholder API keys for providers that need real ones."""
    if settings.llm.provider not in _NO_KEY_PROVIDERS:
        # api_key is typed as plain str. If migrated to SecretStr, use
        # settings.llm.api_key.get_secret_value() instead.
        if settings.llm.api_key in ("no-key", ""):
            raise ValueError(
                f"SCOUT_LLM_API_KEY is required for provider '{settings.llm.provider}'. "
                f"Set it via environment variable or secrets manager."
            )


def _check_persistence(settings: AppSettings) -> None:
    """Warn about file persistence in containerized environments."""
    is_container = bool(
        os.environ.get("ECS_CONTAINER_METADATA_URI")
        or os.environ.get("KUBERNETES_SERVICE_HOST")
    )
    if is_container and settings.persistence.backend == "file":
        log.warning(
            "SCOUT_PERSISTENCE_BACKEND=file in a container environment. "
            "Data will be lost on container restart. Consider setting SCOUT_PERSISTENCE_BACKEND=s3."
        )


def _check_auth(settings: AppSettings) -> None:
    """Reject auth enabled with no credentials — the API would be fully locked."""
    if settings.auth.enabled and not settings.auth.api_keys and not settings.auth.jwks_url:
        raise ValueError(
            "SCOUT_AUTH_ENABLED=true but no API keys or JWKS URL configured. "
            "All authenticated requests would be rejected. "
            "Set SCOUT_AUTH_API_KEYS or SCOUT_AUTH_JWKS_URL, or disable auth."
        )
