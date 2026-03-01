# Enterprise Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden scout-ai for multi-LOB enterprise deployment across Ameritas by addressing security, domain decoupling, and operational maturity gaps identified in the architecture review.

**Architecture:** Three sequential phases. Phase 1 adds authentication, encryption, and tenant isolation to unblock any production deployment. Phase 2 removes APS-specific assumptions from core paths so non-medical domains work correctly. Phase 3 adds IaC, async indexing, and eval infrastructure for scale.

**Tech Stack:** FastAPI security (python-jose, PyJWT), boto3 (Secrets Manager, KMS, SQS), CDK (TypeScript), OpenTelemetry, pytest

---

## Phase 1: Security (Blocks All Production Deployments)

### Task 1: Add JWT Bearer authentication to all API routes

**Files:**
- Create: `src/scout_ai/api/auth.py`
- Modify: `src/scout_ai/api/app.py:40-51`
- Modify: `src/scout_ai/core/config.py` (add `AuthConfig`)
- Create: `tests/unit/test_api_auth.py`

**Step 1: Add `AuthConfig` to settings**

Add to `src/scout_ai/core/config.py` after the `ObservabilityConfig` class:

```python
class AuthConfig(BaseSettings):
    """API authentication configuration.

    Env vars use ``SCOUT_AUTH_`` prefix::

        export SCOUT_AUTH_ENABLED=true
        export SCOUT_AUTH_JWKS_URL=https://cognito-idp.us-east-2.amazonaws.com/us-east-2_xxx/.well-known/jwks.json
    """

    model_config = {"env_prefix": "SCOUT_AUTH_"}

    enabled: bool = False
    jwks_url: str = ""
    issuer: str = ""
    audience: str = "scout-ai"
    algorithm: str = "RS256"
    tenant_claim: str = "custom:tenant_id"
    api_key_header: str = "X-API-Key"
    api_keys: list[str] = Field(default_factory=list)
```

Add `auth: AuthConfig = AuthConfig()` to `AppSettings`.

**Step 2: Write failing tests**

```python
# tests/unit/test_api_auth.py
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def auth_enabled_app():
    """Create app with auth enabled but no valid keys configured."""
    import os
    os.environ["SCOUT_AUTH_ENABLED"] = "true"
    os.environ["SCOUT_AUTH_API_KEYS"] = '["test-key-123"]'
    from importlib import reload
    from scout_ai.api import app as app_module
    reload(app_module)
    yield app_module.app
    os.environ.pop("SCOUT_AUTH_ENABLED", None)
    os.environ.pop("SCOUT_AUTH_API_KEYS", None)


def test_unauthenticated_request_rejected(auth_enabled_app):
    client = TestClient(auth_enabled_app)
    response = client.get("/health")
    assert response.status_code == 200  # health excluded from auth

    response = client.post("/api/index", json={"doc_id": "x", "doc_name": "x", "pages": []})
    assert response.status_code == 401


def test_api_key_auth_accepted(auth_enabled_app):
    client = TestClient(auth_enabled_app)
    response = client.post(
        "/api/index",
        json={"doc_id": "x", "doc_name": "x", "pages": []},
        headers={"X-API-Key": "test-key-123"},
    )
    # Should pass auth (may fail on business logic, but not 401)
    assert response.status_code != 401


def test_auth_disabled_allows_all():
    """When auth is disabled, all requests pass through."""
    from scout_ai.api.app import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_api_auth.py -v`
Expected: FAIL — `auth` module does not exist yet

**Step 4: Implement the auth dependency**

Create `src/scout_ai/api/auth.py`:

```python
"""API authentication: API key and JWT Bearer support."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

if TYPE_CHECKING:
    from scout_ai.core.config import AuthConfig

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_scheme = HTTPBearer(auto_error=False)


def _get_auth_config(request: Request) -> AuthConfig:
    return request.app.state.settings.auth


async def require_auth(
    request: Request,
    api_key: str | None = Security(_api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> str | None:
    """Validate authentication. Returns tenant_id if available."""
    config = _get_auth_config(request)

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
    """Validate JWT and extract tenant claim. Returns tenant_id."""
    try:
        import jwt as pyjwt
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="JWT auth requires PyJWT. Install with: pip install PyJWT[crypto]",
        )

    try:
        payload = pyjwt.decode(
            token,
            options={"verify_signature": False} if not config.jwks_url else {},
            algorithms=[config.algorithm],
            audience=config.audience if config.audience else None,
            issuer=config.issuer if config.issuer else None,
        )
        return payload.get(config.tenant_claim)
    except pyjwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
```

**Step 5: Wire auth into protected routes**

Modify `src/scout_ai/api/app.py` — add auth dependency to protected routers:

```python
from scout_ai.api.auth import require_auth

# Health routes are public (liveness/readiness probes)
app.include_router(health.router)

# All API routes require authentication
app.include_router(index.router, prefix="/api", dependencies=[Depends(require_auth)])
app.include_router(retrieve.router, prefix="/api", dependencies=[Depends(require_auth)])
app.include_router(extract.router, prefix="/api", dependencies=[Depends(require_auth)])
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_api_auth.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/scout_ai/api/auth.py src/scout_ai/api/app.py src/scout_ai/core/config.py tests/unit/test_api_auth.py
git commit -m "feat(security): add JWT + API key authentication to all API routes"
```

---

### Task 2: Add tenant namespace to persistence keys

**Files:**
- Modify: `src/scout_ai/core/config.py` (add `tenant_id` to `AppSettings`)
- Modify: `src/scout_ai/persistence/s3_backend.py:14-28`
- Modify: `src/scout_ai/persistence/file_backend.py`
- Create: `tests/unit/test_tenant_isolation.py`

**Step 1: Add `tenant_id` to AppSettings**

In `src/scout_ai/core/config.py`, add to `AppSettings`:

```python
class AppSettings(BaseSettings):
    domain: str = "aps"
    tenant_id: str = Field(
        default="default",
        description="Tenant/LOB identifier. Prefixed to all persistence keys for isolation.",
    )
    lob: str = Field(
        default="*",
        description="Line of business. Cascades to prompt and rules defaults if they are not set.",
    )
    # ... rest unchanged
```

**Step 2: Write failing test**

```python
# tests/unit/test_tenant_isolation.py
from __future__ import annotations

from scout_ai.persistence.memory_backend import MemoryPersistenceBackend


def test_tenant_prefix_applied():
    """Keys should be prefixed with tenant_id."""
    backend = MemoryPersistenceBackend()
    # Direct access shows raw key
    backend.save("tenant-a/doc1", '{"data": 1}')
    assert backend.exists("tenant-a/doc1")
    assert not backend.exists("doc1")


def test_s3_backend_uses_tenant_prefix():
    """S3 backend should inject tenant_id into key prefix."""
    from scout_ai.persistence.s3_backend import S3PersistenceBackend
    backend = S3PersistenceBackend.__new__(S3PersistenceBackend)
    backend._bucket = "test"
    backend._prefix = "rp/indexes/"
    assert backend._full_key("doc1") == "rp/indexes/doc1.json"
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/unit/test_tenant_isolation.py -v`
Expected: First test PASS (memory backend is simple), second may need adjustment

**Step 4: Modify S3 backend to accept tenant_id**

In `src/scout_ai/persistence/s3_backend.py`, update `__init__`:

```python
def __init__(
    self,
    bucket: str,
    prefix: str = "indexes/",
    region: str = "us-east-2",
    tenant_id: str = "",
    kms_key_id: str = "",
) -> None:
    try:
        import boto3 as _boto3
    except ImportError as e:
        raise ImportError(
            "boto3 is required for S3 persistence. "
            "Install with: pip install scout-ai[s3]"
        ) from e

    self._bucket = bucket
    # Inject tenant prefix: {tenant_id}/{prefix} or just {prefix}
    self._prefix = f"{tenant_id}/{prefix}" if tenant_id else prefix
    self._kms_key_id = kms_key_id
    self._s3 = _boto3.client("s3", region_name=region)
```

**Step 5: Add SSE-KMS to `save()`**

In `src/scout_ai/persistence/s3_backend.py`, update `save()`:

```python
def save(self, key: str, data: str) -> None:
    put_kwargs: dict[str, Any] = {
        "Bucket": self._bucket,
        "Key": self._full_key(key),
        "Body": data.encode("utf-8"),
        "ContentType": "application/json",
    }
    if self._kms_key_id:
        put_kwargs["ServerSideEncryption"] = "aws:kms"
        put_kwargs["SSEKMSKeyId"] = self._kms_key_id
    self._s3.put_object(**put_kwargs)
    log.debug("Saved %s to s3://%s/%s", key, self._bucket, self._full_key(key))
```

**Step 6: Run tests**

Run: `pytest tests/unit/test_tenant_isolation.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/scout_ai/core/config.py src/scout_ai/persistence/s3_backend.py tests/unit/test_tenant_isolation.py
git commit -m "feat(security): add tenant namespace to persistence keys + SSE-KMS support"
```

---

### Task 3: Add SSE-KMS config and PersistenceConfig updates

**Files:**
- Modify: `src/scout_ai/core/config.py` (`PersistenceConfig`)
- Modify: `src/scout_ai/persistence/file_backend.py` (add tenant prefix)

**Step 1: Add KMS and tenant fields to PersistenceConfig**

```python
class PersistenceConfig(BaseSettings):
    model_config = {"env_prefix": "SCOUT_PERSISTENCE_"}

    backend: Literal["file", "s3", "memory"] = "file"
    store_path: Path = Path("./indexes")
    s3_bucket: str = ""
    s3_prefix: str = "indexes/"
    s3_kms_key_id: str = ""
    s3_region: str = ""  # Falls back to top-level aws_region
```

**Step 2: Commit**

```bash
git add src/scout_ai/core/config.py
git commit -m "feat(security): add KMS key ID and region to PersistenceConfig"
```

---

### Task 4: Harden RHEL install script and secrets

**Files:**
- Modify: `deploy/rhel/install.sh`
- Create: `src/scout_ai/core/startup_checks.py`
- Create: `tests/unit/test_startup_checks.py`

**Step 1: Write failing test**

```python
# tests/unit/test_startup_checks.py
from __future__ import annotations

import pytest


def test_rejects_no_key_for_openai_provider():
    from scout_ai.core.startup_checks import validate_settings
    from scout_ai.core.config import AppSettings, LLMConfig

    settings = AppSettings(llm=LLMConfig(provider="openai", api_key="no-key"))
    with pytest.raises(ValueError, match="API key"):
        validate_settings(settings)


def test_accepts_no_key_for_bedrock():
    from scout_ai.core.startup_checks import validate_settings
    from scout_ai.core.config import AppSettings, LLMConfig

    settings = AppSettings(llm=LLMConfig(provider="bedrock", api_key="no-key"))
    validate_settings(settings)  # Should not raise


def test_accepts_no_key_for_ollama():
    from scout_ai.core.startup_checks import validate_settings
    from scout_ai.core.config import AppSettings, LLMConfig

    settings = AppSettings(llm=LLMConfig(provider="ollama", api_key="no-key"))
    validate_settings(settings)  # Should not raise
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_startup_checks.py -v`
Expected: FAIL — module does not exist

**Step 3: Implement startup checks**

Create `src/scout_ai/core/startup_checks.py`:

```python
"""Startup validation: fail-fast for misconfigurations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scout_ai.core.config import AppSettings

log = logging.getLogger(__name__)

# Providers that use IAM/local auth and do not need an API key
_NO_KEY_PROVIDERS = frozenset({"bedrock", "ollama"})


def validate_settings(settings: AppSettings) -> None:
    """Validate settings at startup. Raises ValueError on fatal misconfig."""
    _check_api_key(settings)
    _check_persistence(settings)


def _check_api_key(settings: AppSettings) -> None:
    if settings.llm.provider not in _NO_KEY_PROVIDERS:
        if settings.llm.api_key in ("no-key", ""):
            raise ValueError(
                f"SCOUT_LLM_API_KEY is required for provider '{settings.llm.provider}'. "
                f"Set it via environment variable or secrets manager."
            )


def _check_persistence(settings: AppSettings) -> None:
    import os
    is_container = bool(
        os.environ.get("ECS_CONTAINER_METADATA_URI")
        or os.environ.get("KUBERNETES_SERVICE_HOST")
    )
    if is_container and settings.persistence.backend == "file":
        log.warning(
            "SCOUT_PERSISTENCE_BACKEND=file in a container environment. "
            "Data will be lost on container restart. Consider setting SCOUT_PERSISTENCE_BACKEND=s3."
        )
```

**Step 4: Wire into app lifespan**

In `src/scout_ai/api/app.py`, add to `lifespan()` after `settings = AppSettings()`:

```python
from scout_ai.core.startup_checks import validate_settings
validate_settings(settings)
```

**Step 5: Harden RHEL install script**

In `deploy/rhel/install.sh`, add after directory creation:

```bash
# Create env file with secure permissions
touch "${INSTALL_DIR}/.env"
chmod 600 "${INSTALL_DIR}/.env"
chown scout:scout "${INSTALL_DIR}/.env"
echo "# Configure SCOUT_* environment variables here" > "${INSTALL_DIR}/.env"
```

**Step 6: Run tests**

Run: `pytest tests/unit/test_startup_checks.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/scout_ai/core/startup_checks.py src/scout_ai/api/app.py deploy/rhel/install.sh tests/unit/test_startup_checks.py
git commit -m "feat(security): add startup validation + harden RHEL .env permissions"
```

---

### Task 5: Add OTLP TLS and tenant dimensions to tracing

**Files:**
- Modify: `src/scout_ai/core/config.py` (`ObservabilityConfig`)
- Modify: `src/scout_ai/hooks/tracing.py`
- Create: `tests/unit/test_tracing_config.py`

**Step 1: Add `otlp_insecure` flag to config**

In `ObservabilityConfig`:

```python
class ObservabilityConfig(BaseSettings):
    model_config = {"env_prefix": "SCOUT_OBSERVABILITY_"}

    enable_tracing: bool = False
    otlp_endpoint: str = "http://localhost:4317"
    otlp_insecure: bool = True  # Set to False in production
    service_name: str = "scout-ai"
    log_level: str = "INFO"
```

**Step 2: Update `setup_tracing()` with tenant dimensions and TLS support**

In `src/scout_ai/hooks/tracing.py`:

```python
def setup_tracing(config: ObservabilityConfig, tenant_id: str = "", lob: str = "") -> None:
    if not config.enable_tracing:
        log.debug("Tracing disabled — skipping OpenTelemetry setup")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning(
            "OpenTelemetry packages not found. "
            "Install with: pip install scout-ai[otel]"
        )
        return

    resource_attrs: dict[str, str] = {
        "service.name": config.service_name,
    }
    if tenant_id:
        resource_attrs["service.namespace"] = tenant_id
    if lob:
        resource_attrs["scout.lob"] = lob

    resource = Resource.create(resource_attrs)
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=config.otlp_endpoint,
        insecure=config.otlp_insecure,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    log.info(
        "OpenTelemetry tracing enabled — exporting to %s as '%s' (insecure=%s)",
        config.otlp_endpoint,
        config.service_name,
        config.otlp_insecure,
    )
```

**Step 3: Update lifespan call**

In `src/scout_ai/api/app.py`, update the `setup_tracing` call:

```python
setup_tracing(settings.observability, tenant_id=settings.tenant_id, lob=settings.lob)
```

**Step 4: Commit**

```bash
git add src/scout_ai/core/config.py src/scout_ai/hooks/tracing.py src/scout_ai/api/app.py
git commit -m "feat(security): add TLS support and tenant dimensions to OTLP tracing"
```

---

## Phase 2: Domain Decoupling (Blocks Non-APS LOBs)

### Task 6: Add `SCOUT_AWS_REGION` top-level with cascade

**Files:**
- Modify: `src/scout_ai/core/config.py`
- Create: `tests/unit/test_region_cascade.py`

**Step 1: Write failing test**

```python
# tests/unit/test_region_cascade.py
from __future__ import annotations


def test_top_level_region_cascades_to_persistence():
    """When s3_region is empty, it should fall back to top-level aws_region."""
    from scout_ai.core.config import AppSettings
    settings = AppSettings(aws_region="eu-west-1")
    effective = settings.persistence.s3_region or settings.aws_region
    assert effective == "eu-west-1"


def test_subsystem_region_overrides_top_level():
    """When s3_region is explicitly set, it wins."""
    from scout_ai.core.config import AppSettings, PersistenceConfig
    settings = AppSettings(
        aws_region="eu-west-1",
        persistence=PersistenceConfig(s3_region="us-east-2"),
    )
    effective = settings.persistence.s3_region or settings.aws_region
    assert effective == "us-east-2"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_region_cascade.py -v`
Expected: FAIL — `aws_region` not on `AppSettings`

**Step 3: Add `aws_region` to AppSettings**

In `src/scout_ai/core/config.py`, add to `AppSettings`:

```python
class AppSettings(BaseSettings):
    domain: str = "aps"
    tenant_id: str = "default"
    lob: str = "*"
    aws_region: str = Field(
        default="us-east-2",
        description="Default AWS region. Cascades to subsystems when their region is not set.",
    )
    # ... rest of sub-configs
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_region_cascade.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/scout_ai/core/config.py tests/unit/test_region_cascade.py
git commit -m "feat(config): add SCOUT_AWS_REGION top-level with cascade to subsystems"
```

---

### Task 7: Add `SCOUT_LOB` cascade to prompt and rules defaults

**Files:**
- Modify: `src/scout_ai/api/app.py` (lifespan: cascade lob)
- Create: `tests/unit/test_lob_cascade.py`

**Step 1: Write failing test**

```python
# tests/unit/test_lob_cascade.py
from __future__ import annotations


def test_lob_cascades_to_prompt_default():
    """When SCOUT_LOB is set, it should populate prompt default_lob."""
    from scout_ai.core.config import AppSettings
    settings = AppSettings(lob="retirement_plans")
    # The effective lob for prompts should be the top-level lob
    assert settings.lob == "retirement_plans"
```

**Step 2: Run test**

Run: `pytest tests/unit/test_lob_cascade.py -v`
Expected: PASS (lob field already on AppSettings from Task 2)

**Step 3: Wire into lifespan**

In `src/scout_ai/api/app.py`, update `configure_prompts` call to use `settings.lob` as fallback:

```python
configure_prompts(
    backend=settings.prompt.backend,
    table_name=settings.prompt.table_name,
    aws_region=settings.prompt.aws_region or settings.aws_region,
    cache_ttl_seconds=settings.prompt.cache_ttl_seconds,
    cache_max_size=settings.prompt.cache_max_size,
    fallback_to_file=settings.prompt.fallback_to_file,
    default_lob=settings.prompt.default_lob if settings.prompt.default_lob != "*" else settings.lob,
    default_department=settings.prompt.default_department,
    default_use_case=settings.prompt.default_use_case,
    default_process=settings.prompt.default_process,
)
```

**Step 4: Commit**

```bash
git add src/scout_ai/api/app.py tests/unit/test_lob_cascade.py
git commit -m "feat(config): cascade SCOUT_LOB to prompt and rules defaults"
```

---

### Task 8: Parameterize agent system prompts by domain

**Files:**
- Modify: `src/scout_ai/agents/indexing_agent.py`
- Modify: `src/scout_ai/agents/retrieval_agent.py`
- Modify: `src/scout_ai/agents/extraction_agent.py`
- Create: `src/scout_ai/prompts/templates/base/agent_prompts.py` (domain-neutral fallbacks)
- Create: `tests/unit/test_agent_prompt_routing.py`

**Step 1: Write failing test**

```python
# tests/unit/test_agent_prompt_routing.py
from __future__ import annotations

from unittest.mock import patch


def test_indexing_agent_uses_domain_prompt():
    """Indexing agent should resolve prompt from settings.domain, not hardcoded 'base'."""
    from scout_ai.core.config import AppSettings

    settings = AppSettings(domain="workers_comp")

    with patch("scout_ai.prompts.registry.get_prompt") as mock_get:
        mock_get.return_value = "You are a workers comp indexing specialist."
        from scout_ai.agents.indexing_agent import _resolve_system_prompt
        result = _resolve_system_prompt(settings.domain)
        mock_get.assert_called_once_with(settings.domain, "indexing_agent", "INDEXING_SYSTEM_PROMPT")
        assert "workers comp" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agent_prompt_routing.py -v`
Expected: FAIL — `_resolve_system_prompt` does not exist

**Step 3: Create domain-neutral base prompts**

Create `src/scout_ai/prompts/templates/base/agent_prompts.py`:

```python
"""Domain-neutral agent system prompts — used as fallback when no domain-specific prompt exists."""

from __future__ import annotations

_PROMPT_DATA: dict[str, str] = {
    "INDEXING_SYSTEM_PROMPT": """You are an expert document structure analyst.

Your task is to build a hierarchical tree index from pre-OCR'd document pages. You have tools for \
each step of the indexing pipeline:

1. **build_index**: Start here. Provides the pipeline plan and page metadata.
2. **detect_toc**: Scan the first pages for a table of contents.
3. **process_toc**: Process pages into a structured TOC using the appropriate mode.
4. **verify_toc**: Check that section titles appear on their assigned pages.
5. **fix_incorrect_toc**: Fix entries that failed verification.
6. **split_large_nodes**: Identify and subdivide nodes exceeding size thresholds.
7. **enrich_nodes**: Add summaries, classification, and document description.

Always verify your results and use the fallback cascade: if accuracy is below 60%, try the next \
simpler processing mode.""",

    "RETRIEVAL_SYSTEM_PROMPT": """You are a document retrieval specialist. You search hierarchical \
document tree indexes to find the most relevant sections for extraction questions.

You have two retrieval tools:

1. **tree_search**: Search the tree for a single query. Analyzes the tree structure \
(titles, summaries, content types, page ranges) to identify relevant nodes.

2. **batch_retrieve**: Group extraction questions by their categories and \
run one efficient search per category instead of per question.

Always return node_ids with reasoning explaining your selection.""",

    "EXTRACTION_SYSTEM_PROMPT": """You are a data extraction specialist. You extract precise, \
verifiable answers from document context.

You have two extraction tools:

1. **extract_batch**: For simple lookup questions. Batches up to 20 questions per prompt.

2. **extract_individual**: For complex questions requiring cross-referencing. \
Each question gets its own prompt with step-by-step reasoning.

Critical rules:
- Answer ONLY from the provided context. Never fabricate information.
- If an answer is not found, explicitly say "Not found" with confidence 0.0.
- Every answer MUST include citations with exact page numbers and verbatim quotes.
- Confidence: 1.0 for explicit matches, 0.5-0.8 for inferred, 0.0 for not found.""",
}

_PROMPT_NAMES = frozenset(_PROMPT_DATA.keys())


def __getattr__(name: str) -> str:
    if name in _PROMPT_NAMES:
        from scout_ai.prompts.registry import get_prompt
        return get_prompt("base", "agent", name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(_PROMPT_NAMES) + ["_PROMPT_DATA"]
```

**Step 4: Add `_resolve_system_prompt` helper to each agent factory**

Modify `src/scout_ai/agents/indexing_agent.py`:

```python
"""Indexing agent: builds hierarchical tree indexes from document pages."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from strands import Agent

from scout_ai.agents.factory import create_model
from scout_ai.hooks import AuditHook, CostHook
from scout_ai.skills.indexing import (
    build_index, detect_toc, enrich_nodes, fix_incorrect_toc,
    process_toc, split_large_nodes, verify_toc,
)

if TYPE_CHECKING:
    from scout_ai.core.config import AppSettings

log = logging.getLogger(__name__)


def _resolve_system_prompt(domain: str) -> str:
    """Resolve system prompt from domain, falling back to base."""
    from scout_ai.prompts.registry import get_prompt
    try:
        return get_prompt(domain, "indexing_agent", "INDEXING_SYSTEM_PROMPT")
    except KeyError:
        log.debug("No domain-specific indexing prompt for %r, using base", domain)
        return get_prompt("base", "agent", "INDEXING_SYSTEM_PROMPT")


def create_indexing_agent(settings: AppSettings, **kwargs: Any) -> Agent:
    model = create_model(settings)
    system_prompt = _resolve_system_prompt(settings.domain)

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            build_index, detect_toc, process_toc, verify_toc,
            fix_incorrect_toc, split_large_nodes, enrich_nodes,
        ],
        hooks=[AuditHook(), CostHook()],
        trace_attributes={"agent.type": "indexing", "agent.domain": settings.domain},
        name="Scout Indexing Agent",
        description="Builds hierarchical tree indexes from pre-OCR'd document pages",
        **kwargs,
    )
```

Apply the same pattern to `retrieval_agent.py` and `extraction_agent.py`, changing the prompt names accordingly:

- retrieval: `get_prompt(domain, "retrieval_agent", "RETRIEVAL_SYSTEM_PROMPT")`
- extraction: `get_prompt(domain, "extraction_agent", "EXTRACTION_SYSTEM_PROMPT")`

**Step 5: Run tests**

Run: `pytest tests/unit/test_agent_prompt_routing.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/scout_ai/prompts/templates/base/agent_prompts.py \
  src/scout_ai/agents/indexing_agent.py \
  src/scout_ai/agents/retrieval_agent.py \
  src/scout_ai/agents/extraction_agent.py \
  tests/unit/test_agent_prompt_routing.py
git commit -m "feat(domains): parameterize agent system prompts by settings.domain with base fallback"
```

---

### Task 9: Fix `run_with_synthesis()` APS fallback to use domain registry

**Files:**
- Modify: `src/scout_ai/agents/orchestrator.py:160-169`
- Create: `tests/unit/test_orchestrator_synthesis_routing.py`

**Step 1: Write failing test**

```python
# tests/unit/test_orchestrator_synthesis_routing.py
from __future__ import annotations

import pytest


def test_run_with_synthesis_rejects_unknown_domain():
    """Should raise ValueError for domain with no synthesis_pipeline, not silently use APS."""
    from scout_ai.agents.orchestrator import ExtractionPipeline
    from unittest.mock import AsyncMock

    pipeline = ExtractionPipeline(
        retrieval_provider=AsyncMock(),
        chat_provider=AsyncMock(),
    )

    # Mock the domain registry to return a config with no synthesis_pipeline
    from scout_ai.domains.registry import DomainConfig, DomainRegistry
    with pytest.MonkeyPatch.context() as mp:
        reg = DomainRegistry()
        reg.register(DomainConfig(name="test_domain", display_name="Test"))
        mp.setattr("scout_ai.agents.orchestrator.get_registry", lambda: reg)
        # Should not silently fall back to APS
        # (This test validates the behavior after the fix)
```

**Step 2: Fix the APS fallback**

In `src/scout_ai/agents/orchestrator.py`, replace lines 160-169 in `run_with_synthesis`:

```python
# BEFORE (broken):
if synthesis_pipeline is None:
    from scout_ai.domains.aps.synthesis.pipeline import SynthesisPipeline
    # ... APS hardcode

# AFTER (domain-routed):
if synthesis_pipeline is None:
    from scout_ai.domains.registry import get_registry
    try:
        domain_config = get_registry().get(settings_domain)
        SynthCls = domain_config.resolve("synthesis_pipeline")
    except (KeyError, ValueError) as e:
        log.warning("No synthesis pipeline for domain %r: %s", settings_domain, e)
        return batch_results, None, None

    cache_enabled = getattr(self._chat, "_cache_enabled", False)
    client = getattr(self._chat, "_client", None)
    if client is None:
        log.warning("Cannot synthesize: chat provider has no _client attribute")
        return batch_results, None, None
    synthesis_pipeline = SynthCls(client, cache_enabled=cache_enabled)
```

Note: `run_with_synthesis` will need a `settings_domain: str = "aps"` parameter added to its signature, or the `ExtractionPipeline.__init__` should accept `domain: str`.

**Step 3: Add `domain` to ExtractionPipeline**

```python
class ExtractionPipeline:
    def __init__(
        self,
        retrieval_provider: IRetrievalProvider,
        chat_provider: IChatProvider,
        domain: str = "aps",
    ) -> None:
        self._retrieval = retrieval_provider
        self._chat = chat_provider
        self._domain = domain
```

Then use `self._domain` instead of hardcoded APS import.

**Step 4: Run tests**

Run: `pytest tests/unit/test_orchestrator_synthesis_routing.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/scout_ai/agents/orchestrator.py tests/unit/test_orchestrator_synthesis_routing.py
git commit -m "fix(domains): route synthesis pipeline through domain registry instead of hardcoded APS"
```

---

### Task 10: Parameterize legacy indexer prompts by domain

**Files:**
- Modify: `src/scout_ai/providers/pageindex/indexer.py:39-50`
- Create: `tests/unit/test_indexer_prompt_routing.py`

**Step 1: Write failing test**

```python
# tests/unit/test_indexer_prompt_routing.py
from __future__ import annotations

from unittest.mock import patch


def test_default_prompt_uses_domain_not_hardcoded_aps():
    """_default_prompt should accept domain parameter and use it for lookup."""
    with patch("scout_ai.prompts.registry.get_prompt") as mock_get:
        mock_get.return_value = "test prompt"
        from scout_ai.providers.pageindex.indexer import _default_prompt
        result = _default_prompt("TOC_DETECT_PROMPT", domain="workers_comp")
        mock_get.assert_called_with("workers_comp", "indexing", "TOC_DETECT_PROMPT")
```

**Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_indexer_prompt_routing.py -v`
Expected: FAIL — `_default_prompt` does not accept `domain`

**Step 3: Fix `_default_prompt` to accept domain**

In `src/scout_ai/providers/pageindex/indexer.py`, change:

```python
def _default_prompt(name: str, domain: str = "aps") -> str:
    """Lazy-load a prompt from the domain registry with base fallback."""
    from scout_ai.prompts.registry import get_prompt

    try:
        return get_prompt(domain, "indexing", name)
    except KeyError:
        return get_prompt("base", "indexing", name)
```

Update all call sites within the file to pass `self._domain` (add `domain` to `ScoutIndexer.__init__` if not already present).

**Step 4: Run tests**

Run: `pytest tests/unit/test_indexer_prompt_routing.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/scout_ai/providers/pageindex/indexer.py tests/unit/test_indexer_prompt_routing.py
git commit -m "feat(domains): parameterize legacy indexer prompts by domain"
```

---

### Task 11: Add per-stage model configuration

**Files:**
- Modify: `src/scout_ai/core/config.py` (add `StageModelConfig`)
- Modify: `src/scout_ai/agents/factory.py` (add `model_override` parameter)
- Create: `tests/unit/test_stage_model_config.py`

**Step 1: Write failing test**

```python
# tests/unit/test_stage_model_config.py
from __future__ import annotations


def test_stage_model_overrides_default():
    from scout_ai.core.config import AppSettings, StageModelConfig
    settings = AppSettings(
        stage_models=StageModelConfig(retrieval_model="claude-3-haiku-20240307"),
    )
    effective = settings.stage_models.retrieval_model or settings.llm.model
    assert effective == "claude-3-haiku-20240307"


def test_stage_model_falls_back_to_default():
    from scout_ai.core.config import AppSettings, StageModelConfig
    settings = AppSettings(
        stage_models=StageModelConfig(),  # all empty
    )
    effective = settings.stage_models.retrieval_model or settings.llm.model
    assert effective == settings.llm.model
```

**Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_stage_model_config.py -v`
Expected: FAIL — `StageModelConfig` does not exist

**Step 3: Add StageModelConfig**

In `src/scout_ai/core/config.py`:

```python
class StageModelConfig(BaseSettings):
    """Per-pipeline-stage model overrides.

    Env vars use ``SCOUT_STAGE_`` prefix::

        export SCOUT_STAGE_RETRIEVAL_MODEL=claude-3-haiku-20240307
        export SCOUT_STAGE_EXTRACTION_MODEL=claude-3-5-sonnet-20241022
    """

    model_config = {"env_prefix": "SCOUT_STAGE_"}

    indexing_model: str = ""
    retrieval_model: str = ""
    extraction_model: str = ""
    synthesis_model: str = ""
```

Add `stage_models: StageModelConfig = StageModelConfig()` to `AppSettings`.

**Step 4: Add `model_override` to factory**

In `src/scout_ai/agents/factory.py`:

```python
def create_model(settings: AppSettings, model_override: str = "") -> Model:
    """Instantiate a Strands model. If model_override is provided, use it instead of settings.llm.model."""
    provider = settings.llm.provider
    effective_model = model_override or settings.llm.model
    # Use effective_model in all provider branches instead of settings.llm.model
    ...
```

**Step 5: Wire into agent factories**

In each agent factory (indexing, retrieval, extraction), pass the stage override:

```python
# In create_retrieval_agent:
model = create_model(settings, model_override=settings.stage_models.retrieval_model)
```

**Step 6: Run tests**

Run: `pytest tests/unit/test_stage_model_config.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/scout_ai/core/config.py src/scout_ai/agents/factory.py \
  src/scout_ai/agents/indexing_agent.py src/scout_ai/agents/retrieval_agent.py \
  src/scout_ai/agents/extraction_agent.py tests/unit/test_stage_model_config.py
git commit -m "feat(config): add per-stage model overrides via SCOUT_STAGE_* env vars"
```

---

## Phase 3: Operational Maturity (Blocks Scale)

### Task 12: Create CDK stack per LOB

**Files:**
- Create: `infra/cdk/bin/app.ts`
- Create: `infra/cdk/lib/scout-stack.ts`
- Create: `infra/cdk/package.json`
- Create: `infra/cdk/tsconfig.json`
- Create: `infra/cdk/cdk.json`

**Step 1: Initialize CDK project**

```bash
mkdir -p infra/cdk && cd infra/cdk
npx cdk init app --language typescript
```

**Step 2: Implement the ScoutStack**

Create `infra/cdk/lib/scout-stack.ts`:

```typescript
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ecs_patterns from 'aws-cdk-lib/aws-ecs-patterns';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import { Construct } from 'constructs';

interface ScoutStackProps extends cdk.StackProps {
  lob: string;        // e.g., "rp", "individual", "group-health"
  domain: string;     // e.g., "aps", "workers_comp"
  environment: string; // e.g., "dev", "staging", "prod"
}

export class ScoutStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: ScoutStackProps) {
    super(scope, id, props);

    const { lob, domain, environment } = props;
    const prefix = `scout-${lob}-${environment}`;

    // VPC with private subnets
    const vpc = new ec2.Vpc(this, 'Vpc', {
      maxAzs: 2,
      natGateways: 1,
    });

    // S3 bucket with SSE-KMS and lifecycle
    const indexBucket = new s3.Bucket(this, 'IndexBucket', {
      bucketName: `${prefix}-indexes`,
      encryption: s3.BucketEncryption.KMS_MANAGED,
      enforceSSL: true,
      versioned: true,
      lifecycleRules: [{
        transitions: [{ storageClass: s3.StorageClass.INFREQUENT_ACCESS, transitionAfter: cdk.Duration.days(90) }],
        expiration: cdk.Duration.days(365),
      }],
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // DynamoDB tables
    const promptTable = new dynamodb.Table(this, 'PromptTable', {
      tableName: `${prefix}-prompts`,
      partitionKey: { name: 'PK', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'SK', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      pointInTimeRecovery: true,
      encryption: dynamodb.TableEncryption.AWS_MANAGED,
    });

    const rulesTable = new dynamodb.Table(this, 'RulesTable', {
      tableName: `${prefix}-rules`,
      partitionKey: { name: 'PK', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'SK', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      pointInTimeRecovery: true,
    });

    // SQS queue for async indexing
    const indexDlq = new sqs.Queue(this, 'IndexDLQ', {
      queueName: `${prefix}-index-dlq`,
      retentionPeriod: cdk.Duration.days(14),
    });

    const indexQueue = new sqs.Queue(this, 'IndexQueue', {
      queueName: `${prefix}-index-queue`,
      visibilityTimeout: cdk.Duration.minutes(15),
      deadLetterQueue: { queue: indexDlq, maxReceiveCount: 3 },
    });

    // Secrets Manager for API keys
    const apiKeySecret = new secretsmanager.Secret(this, 'ApiKeySecret', {
      secretName: `${prefix}/llm-api-key`,
      description: `LLM API key for Scout AI ${lob} ${environment}`,
    });

    // ECS Cluster
    const cluster = new ecs.Cluster(this, 'Cluster', {
      vpc,
      clusterName: prefix,
      containerInsights: true,
    });

    // Fargate Service with ALB
    const service = new ecs_patterns.ApplicationLoadBalancedFargateService(this, 'Service', {
      cluster,
      cpu: 1024,
      memoryLimitMiB: 2048,
      desiredCount: 2,
      taskImageOptions: {
        image: ecs.ContainerImage.fromRegistry(`${this.account}.dkr.ecr.${this.region}.amazonaws.com/scout-ai:latest`),
        containerPort: 8080,
        environment: {
          SCOUT_DOMAIN: domain,
          SCOUT_TENANT_ID: lob,
          SCOUT_LOB: lob,
          SCOUT_AWS_REGION: this.region,
          SCOUT_PERSISTENCE_BACKEND: 's3',
          SCOUT_PERSISTENCE_S3_BUCKET: indexBucket.bucketName,
          SCOUT_PERSISTENCE_S3_PREFIX: `${lob}/indexes/`,
          SCOUT_PROMPT_BACKEND: 'dynamodb',
          SCOUT_PROMPT_TABLE_NAME: promptTable.tableName,
          SCOUT_RULES_BACKEND: 'dynamodb',
          SCOUT_RULES_TABLE_NAME: rulesTable.tableName,
          SCOUT_AUTH_ENABLED: 'true',
        },
        secrets: {
          SCOUT_LLM_API_KEY: ecs.Secret.fromSecretsManager(apiKeySecret),
        },
      },
      publicLoadBalancer: false,
    });

    // Grant permissions
    indexBucket.grantReadWrite(service.taskDefinition.taskRole);
    promptTable.grantReadData(service.taskDefinition.taskRole);
    rulesTable.grantReadData(service.taskDefinition.taskRole);
    indexQueue.grantSendMessages(service.taskDefinition.taskRole);
    apiKeySecret.grantRead(service.taskDefinition.taskRole);

    // Bedrock access
    service.taskDefinition.taskRole.addToPrincipalPolicy(
      new iam.PolicyStatement({
        actions: ['bedrock:InvokeModel', 'bedrock:InvokeModelWithResponseStream'],
        resources: ['*'],
      })
    );

    // Outputs
    new cdk.CfnOutput(this, 'LoadBalancerDNS', { value: service.loadBalancer.loadBalancerDnsName });
    new cdk.CfnOutput(this, 'S3Bucket', { value: indexBucket.bucketName });
    new cdk.CfnOutput(this, 'IndexQueueUrl', { value: indexQueue.queueUrl });
  }
}
```

**Step 3: Create bin/app.ts**

```typescript
#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { ScoutStack } from '../lib/scout-stack';

const app = new cdk.App();

const lob = app.node.tryGetContext('lob') || 'rp';
const domain = app.node.tryGetContext('domain') || 'aps';
const environment = app.node.tryGetContext('env') || 'dev';

new ScoutStack(app, `Scout-${lob}-${environment}`, {
  lob,
  domain,
  environment,
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || 'us-east-2',
  },
});
```

Usage: `cdk deploy -c lob=rp -c domain=aps -c env=prod`

**Step 4: Commit**

```bash
git add infra/
git commit -m "feat(infra): add CDK stack for per-LOB AWS infrastructure provisioning"
```

---

### Task 13: Add shared CI/CD pipeline template

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/deploy.yml`

**Step 1: Create CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -e ".[dev]"
      - run: ruff check src/ tests/
      - run: mypy src/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --tb=short

  docker:
    runs-on: ubuntu-latest
    needs: [lint-and-type-check, test]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-arn: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: us-east-2
      - uses: aws-actions/amazon-ecr-login@v2
        id: ecr
      - run: |
          docker build -f docker/Dockerfile -t ${{ steps.ecr.outputs.registry }}/scout-ai:${{ github.sha }} .
          docker push ${{ steps.ecr.outputs.registry }}/scout-ai:${{ github.sha }}
          docker tag ${{ steps.ecr.outputs.registry }}/scout-ai:${{ github.sha }} ${{ steps.ecr.outputs.registry }}/scout-ai:latest
          docker push ${{ steps.ecr.outputs.registry }}/scout-ai:latest

  config-schema:
    runs-on: ubuntu-latest
    needs: [test]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -e .
      - run: python -c "from scout_ai.core.config import AppSettings; import json; print(json.dumps(AppSettings.model_json_schema(), indent=2))" > config-schema.json
      - uses: actions/upload-artifact@v4
        with:
          name: config-schema
          path: config-schema.json
```

**Step 2: Commit**

```bash
git add .github/
git commit -m "feat(ci): add shared CI pipeline with lint, test, docker, and config schema"
```

---

### Task 14: Externalize circuit breaker state

**Files:**
- Modify: `src/scout_ai/hooks/circuit_breaker_hook.py`
- Create: `src/scout_ai/hooks/circuit_breaker_store.py`
- Create: `tests/unit/test_circuit_breaker_external.py`

**Step 1: Write failing test**

```python
# tests/unit/test_circuit_breaker_external.py
from __future__ import annotations


def test_memory_store_records_failures():
    from scout_ai.hooks.circuit_breaker_store import MemoryBreakerStore
    store = MemoryBreakerStore()
    store.record_failure("bedrock/claude-3-sonnet")
    assert store.get_failure_count("bedrock/claude-3-sonnet") == 1
    store.record_failure("bedrock/claude-3-sonnet")
    assert store.get_failure_count("bedrock/claude-3-sonnet") == 2


def test_memory_store_reset():
    from scout_ai.hooks.circuit_breaker_store import MemoryBreakerStore
    store = MemoryBreakerStore()
    store.record_failure("bedrock/claude-3-sonnet")
    store.reset("bedrock/claude-3-sonnet")
    assert store.get_failure_count("bedrock/claude-3-sonnet") == 0


def test_circuit_breaker_uses_store():
    from scout_ai.hooks.circuit_breaker_hook import CircuitBreakerHook
    from scout_ai.hooks.circuit_breaker_store import MemoryBreakerStore

    store = MemoryBreakerStore()
    breaker = CircuitBreakerHook(failure_threshold=3, store=store, breaker_key="test")
    assert breaker.state.value == "closed"
```

**Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_circuit_breaker_external.py -v`
Expected: FAIL — `circuit_breaker_store` does not exist

**Step 3: Create the store protocol and implementations**

Create `src/scout_ai/hooks/circuit_breaker_store.py`:

```python
"""Pluggable circuit breaker state stores."""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class IBreakerStore(Protocol):
    """Protocol for circuit breaker external state."""

    def record_failure(self, key: str) -> int:
        """Record a failure and return new count."""
        ...

    def get_failure_count(self, key: str) -> int:
        """Get current consecutive failure count."""
        ...

    def get_last_failure_time(self, key: str) -> float:
        """Get monotonic timestamp of last failure."""
        ...

    def reset(self, key: str) -> None:
        """Reset failure count to zero."""
        ...


class MemoryBreakerStore:
    """In-process store — same behavior as before, for single-replica deployments."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._times: dict[str, float] = {}

    def record_failure(self, key: str) -> int:
        self._counts[key] = self._counts.get(key, 0) + 1
        self._times[key] = time.monotonic()
        return self._counts[key]

    def get_failure_count(self, key: str) -> int:
        return self._counts.get(key, 0)

    def get_last_failure_time(self, key: str) -> float:
        return self._times.get(key, 0.0)

    def reset(self, key: str) -> None:
        self._counts[key] = 0
```

**Step 4: Refactor CircuitBreakerHook to use store**

In `src/scout_ai/hooks/circuit_breaker_hook.py`:

```python
class CircuitBreakerHook:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 60.0,
        store: IBreakerStore | None = None,
        breaker_key: str = "default",
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._key = breaker_key

        if store is None:
            from scout_ai.hooks.circuit_breaker_store import MemoryBreakerStore
            store = MemoryBreakerStore()
        self._store = store
        self._state = CircuitState.CLOSED

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._store.get_last_failure_time(self._key)
            if elapsed >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                log.info("Circuit breaker → HALF_OPEN (recovery timeout elapsed)")
        return self._state

    def _after_model_call(self, event: Any) -> None:
        error = getattr(event, "error", None)
        if error:
            count = self._store.record_failure(self._key)
            if count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                log.warning("Circuit breaker → OPEN after %d failures", count)
        else:
            if self._state == CircuitState.HALF_OPEN:
                log.info("Circuit breaker → CLOSED (successful call in half-open)")
            self._store.reset(self._key)
            self._state = CircuitState.CLOSED

    def reset(self) -> None:
        self._store.reset(self._key)
        self._state = CircuitState.CLOSED
```

**Step 5: Run tests**

Run: `pytest tests/unit/test_circuit_breaker_external.py tests/unit/ -v -k "circuit"`
Expected: PASS (both new and existing tests)

**Step 6: Commit**

```bash
git add src/scout_ai/hooks/circuit_breaker_hook.py src/scout_ai/hooks/circuit_breaker_store.py \
  tests/unit/test_circuit_breaker_external.py
git commit -m "refactor(hooks): externalize circuit breaker state via IBreakerStore protocol"
```

---

### Task 15: Add async indexing via SQS

**Files:**
- Create: `src/scout_ai/api/routes/index_async.py`
- Create: `src/scout_ai/worker/sqs_consumer.py`
- Modify: `src/scout_ai/api/app.py` (include new router)
- Create: `tests/unit/test_async_indexing.py`

**Step 1: Write failing test**

```python
# tests/unit/test_async_indexing.py
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_async_index_returns_job_id():
    """POST /api/index/async should return a job_id immediately."""
    from fastapi.testclient import TestClient
    from scout_ai.api.app import app

    with patch("scout_ai.api.routes.index_async._send_to_queue") as mock_send:
        mock_send.return_value = "job-abc-123"
        client = TestClient(app)
        response = client.post(
            "/api/index/async",
            json={"doc_id": "test", "doc_name": "Test", "pages": []},
        )
        assert response.status_code == 202
        assert "job_id" in response.json()
```

**Step 2: Implement async index route**

Create `src/scout_ai/api/routes/index_async.py`:

```python
"""Async indexing endpoint — enqueues to SQS and returns immediately."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(tags=["indexing"])


class AsyncIndexRequest(BaseModel):
    doc_id: str
    doc_name: str
    pages: list[dict[str, object]] = Field(...)


class AsyncIndexResponse(BaseModel):
    job_id: str
    status: str = "queued"
    message: str = "Indexing job has been queued for processing."


def _send_to_queue(queue_url: str, message: dict, region: str) -> str:
    """Send indexing job to SQS queue. Returns message ID."""
    import boto3
    sqs = boto3.client("sqs", region_name=region)
    job_id = str(uuid.uuid4())
    message["job_id"] = job_id
    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(message),
        MessageGroupId=message.get("doc_id", "default"),
    )
    return job_id


@router.post("/index/async", response_model=AsyncIndexResponse, status_code=202)
async def create_index_async(request: AsyncIndexRequest, req: Request) -> AsyncIndexResponse:
    """Enqueue a document for async indexing."""
    settings = req.app.state.settings
    queue_url = getattr(settings, "index_queue_url", "")
    if not queue_url:
        from fastapi import HTTPException
        raise HTTPException(status_code=501, detail="Async indexing not configured. Set SCOUT_INDEX_QUEUE_URL.")

    job_id = _send_to_queue(
        queue_url=queue_url,
        message=request.model_dump(mode="json"),
        region=settings.aws_region,
    )
    return AsyncIndexResponse(job_id=job_id)
```

**Step 3: Wire into app**

In `src/scout_ai/api/app.py`:

```python
from scout_ai.api.routes import extract, health, index, index_async, retrieve

app.include_router(index_async.router, prefix="/api", dependencies=[Depends(require_auth)])
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_async_indexing.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/scout_ai/api/routes/index_async.py src/scout_ai/api/app.py tests/unit/test_async_indexing.py
git commit -m "feat(scale): add async indexing via SQS with /api/index/async endpoint"
```

---

### Task 16: Build retrieval eval harness

**Files:**
- Create: `tests/eval/__init__.py`
- Create: `tests/eval/golden_dataset.py`
- Create: `tests/eval/retrieval_evaluator.py`
- Create: `tests/eval/test_eval_harness.py`

**Step 1: Create golden dataset model**

Create `tests/eval/golden_dataset.py`:

```python
"""Golden dataset model for retrieval and extraction evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class GoldenRetrievalCase(BaseModel):
    """A single retrieval eval case."""

    doc_id: str
    query: str
    category: str
    expected_node_ids: list[str] = Field(default_factory=list)
    expected_page_range: tuple[int, int] | None = None


class GoldenExtractionCase(BaseModel):
    """A single extraction eval case."""

    doc_id: str
    question_id: str
    category: str
    expected_answer: str
    tolerance: str = "exact"  # exact | contains | numeric_within_5pct


class GoldenDataset(BaseModel):
    """A collection of eval cases for a domain."""

    domain: str
    version: str = "1.0"
    retrieval_cases: list[GoldenRetrievalCase] = Field(default_factory=list)
    extraction_cases: list[GoldenExtractionCase] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> GoldenDataset:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

**Step 2: Create retrieval evaluator**

Create `tests/eval/retrieval_evaluator.py`:

```python
"""Retrieval evaluation metrics: Recall@k and MRR."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievalMetrics:
    """Computed metrics for a set of retrieval cases."""

    total_cases: int = 0
    recall_at_5: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    per_case: list[dict[str, object]] = field(default_factory=list)


def compute_recall_at_k(
    expected_node_ids: list[str],
    retrieved_node_ids: list[str],
    k: int = 5,
) -> float:
    """Proportion of expected nodes found in top-k retrieved."""
    if not expected_node_ids:
        return 1.0
    top_k = set(retrieved_node_ids[:k])
    hits = len(set(expected_node_ids) & top_k)
    return hits / len(expected_node_ids)


def compute_mrr(
    expected_node_ids: list[str],
    retrieved_node_ids: list[str],
) -> float:
    """Reciprocal rank of first relevant result."""
    expected_set = set(expected_node_ids)
    for i, node_id in enumerate(retrieved_node_ids, start=1):
        if node_id in expected_set:
            return 1.0 / i
    return 0.0
```

**Step 3: Write tests for the harness**

Create `tests/eval/test_eval_harness.py`:

```python
"""Tests for the eval harness itself."""

from __future__ import annotations

from tests.eval.retrieval_evaluator import compute_mrr, compute_recall_at_k


def test_recall_at_5_perfect():
    assert compute_recall_at_k(["a", "b"], ["a", "b", "c", "d", "e"]) == 1.0


def test_recall_at_5_partial():
    assert compute_recall_at_k(["a", "b"], ["c", "a", "d", "e", "f"]) == 0.5


def test_recall_at_5_none():
    assert compute_recall_at_k(["a", "b"], ["c", "d", "e", "f", "g"]) == 0.0


def test_mrr_first():
    assert compute_mrr(["a"], ["a", "b", "c"]) == 1.0


def test_mrr_second():
    assert compute_mrr(["b"], ["a", "b", "c"]) == 0.5


def test_mrr_not_found():
    assert compute_mrr(["x"], ["a", "b", "c"]) == 0.0
```

**Step 4: Run tests**

Run: `pytest tests/eval/test_eval_harness.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/eval/
git commit -m "feat(eval): add retrieval eval harness with golden dataset model, Recall@k, and MRR"
```

---

### Task 17: Add feedback capture API endpoint

**Files:**
- Create: `src/scout_ai/api/routes/feedback.py`
- Modify: `src/scout_ai/api/app.py`
- Create: `tests/unit/test_feedback_endpoint.py`

**Step 1: Write failing test**

```python
# tests/unit/test_feedback_endpoint.py
from __future__ import annotations

from fastapi.testclient import TestClient


def test_feedback_endpoint_accepts_correction():
    from scout_ai.api.app import app
    client = TestClient(app)
    response = client.post("/api/feedback", json={
        "doc_id": "doc-123",
        "question_id": "q-1",
        "corrected_answer": "The correct value is 42",
        "auditor_id": "jane.doe",
    })
    assert response.status_code == 201
    assert response.json()["status"] == "recorded"
```

**Step 2: Implement feedback route**

Create `src/scout_ai/api/routes/feedback.py`:

```python
"""Feedback capture endpoint for human-in-the-loop corrections."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    doc_id: str
    question_id: str
    corrected_answer: str
    corrected_citations: list[dict[str, object]] = Field(default_factory=list)
    auditor_id: str = ""
    notes: str = ""


class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str = "recorded"


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(request: FeedbackRequest, req: Request) -> FeedbackResponse:
    """Record a human correction for a previous extraction result."""
    settings = req.app.state.settings

    from scout_ai.persistence import create_persistence_backend
    backend = create_persistence_backend(settings)

    feedback_id = f"{request.doc_id}_{request.question_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    payload = {
        **request.model_dump(mode="json"),
        "feedback_id": feedback_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    backend.save(f"feedback/{feedback_id}", json.dumps(payload))

    return FeedbackResponse(feedback_id=feedback_id)
```

**Step 3: Wire into app**

In `src/scout_ai/api/app.py`:

```python
from scout_ai.api.routes import extract, feedback, health, index, index_async, retrieve

app.include_router(feedback.router, prefix="/api", dependencies=[Depends(require_auth)])
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_feedback_endpoint.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/scout_ai/api/routes/feedback.py src/scout_ai/api/app.py tests/unit/test_feedback_endpoint.py
git commit -m "feat(eval): add /api/feedback endpoint for human-in-the-loop corrections"
```

---

### Task 18: Add LOB/tenant dimensions to OTEL spans in AuditHook

**Files:**
- Modify: `src/scout_ai/hooks/audit_hook.py`
- Create: `tests/unit/test_audit_hook_dimensions.py`

**Step 1: Write failing test**

```python
# tests/unit/test_audit_hook_dimensions.py
from __future__ import annotations


def test_audit_hook_accepts_tenant_dimensions():
    from scout_ai.hooks.audit_hook import AuditHook
    hook = AuditHook(tenant_id="rp", lob="retirement_plans", domain="aps")
    assert hook._tenant_id == "rp"
    assert hook._lob == "retirement_plans"
    assert hook._domain == "aps"
```

**Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_audit_hook_dimensions.py -v`
Expected: FAIL — constructor does not accept these params

**Step 3: Add tenant dimensions to AuditHook**

Modify `src/scout_ai/hooks/audit_hook.py` constructor to accept optional dimensions:

```python
class AuditHook:
    def __init__(
        self,
        tenant_id: str = "",
        lob: str = "",
        domain: str = "",
    ) -> None:
        self._tenant_id = tenant_id
        self._lob = lob
        self._domain = domain
```

Include these in all log entries via the `_on_model_call` and `_on_tool_call` handlers as additional structlog fields.

**Step 4: Run tests**

Run: `pytest tests/unit/test_audit_hook_dimensions.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/scout_ai/hooks/audit_hook.py tests/unit/test_audit_hook_dimensions.py
git commit -m "feat(observability): add tenant/LOB/domain dimensions to AuditHook log entries"
```

---

## Summary: Task Dependency Graph

```
Phase 1 (Security)           Phase 2 (Domain Decoupling)      Phase 3 (Operational Maturity)
┌──────────────────┐         ┌──────────────────────┐         ┌──────────────────────────┐
│ T1: JWT Auth     │         │ T6: AWS Region       │         │ T12: CDK Stack           │
│ T2: Tenant NS    │         │ T7: LOB Cascade      │         │ T13: CI/CD Pipeline      │
│ T3: KMS Config   │         │ T8: Agent Prompts    │         │ T14: Circuit Breaker Ext │
│ T4: Startup Chks │         │ T9: Synthesis Fix    │         │ T15: SQS Async Index     │
│ T5: OTLP TLS     │         │ T10: Indexer Prompts │         │ T16: Eval Harness        │
└──────────────────┘         │ T11: Stage Models    │         │ T17: Feedback API        │
         │                   └──────────────────────┘         │ T18: Audit Dimensions    │
         ▼                            │                       └──────────────────────────┘
   Phase 2 starts                     ▼                                │
                               Phase 3 starts                         ▼
                                                                   Done
```

Tasks within each phase are independent and can be parallelized.
Phase boundaries are sequential: 1 → 2 → 3.
