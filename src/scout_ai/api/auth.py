"""API authentication: API key and JWT Bearer support."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

if TYPE_CHECKING:
    from scout_ai.core.config import AuthConfig

log = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_scheme = HTTPBearer(auto_error=False)


async def require_auth(
    request: Request,
    api_key: str | None = Security(_api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> str | None:
    """Validate authentication. Returns tenant_id if available, None otherwise."""
    config: AuthConfig = request.app.state.settings.auth

    if not config.enabled:
        return None

    # Try API key first
    if api_key and api_key in config.api_keys:
        return None

    # Try JWT Bearer
    if bearer:
        return _validate_jwt(bearer.credentials, config)

    raise HTTPException(
        status_code=401,
        detail="Authentication required. Provide X-API-Key header or Bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _validate_jwt(token: str, config: AuthConfig) -> str | None:
    """Validate JWT and extract tenant claim. Returns tenant_id or None."""
    try:
        import jwt as pyjwt
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="JWT auth requires PyJWT. Install with: pip install PyJWT[crypto]",
        )

    try:
        if config.jwks_url:
            # Full verification: fetch signing key from JWKS endpoint
            jwk_client = pyjwt.PyJWKClient(config.jwks_url)
            signing_key = jwk_client.get_signing_key_from_jwt(token)
            payload = pyjwt.decode(
                token,
                key=signing_key.key,
                algorithms=[config.algorithm],
                audience=config.audience if config.audience else None,
                issuer=config.issuer if config.issuer else None,
            )
        else:
            # Gateway-terminated auth: upstream proxy verified the signature,
            # we only decode claims (audience, issuer, expiry still checked).
            log.warning(
                "JWT signature verification disabled (no JWKS URL). "
                "Ensure requests are proxied through an authenticating gateway."
            )
            payload = pyjwt.decode(
                token,
                options={"verify_signature": False, "verify_exp": True},
                algorithms=[config.algorithm],
                audience=config.audience if config.audience else None,
                issuer=config.issuer if config.issuer else None,
            )
        return payload.get(config.tenant_claim)
    except pyjwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
