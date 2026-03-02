"""FastAPI application with lifespan management."""

from __future__ import annotations

import importlib.metadata
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI

from scout_ai.api.auth import require_auth
from scout_ai.api.routes import extract, feedback, health, index, index_async, retrieve
from scout_ai.core.config import APIConfig, AppSettings
from scout_ai.core.startup_checks import validate_settings
from scout_ai.hooks import setup_logging, setup_tracing
from scout_ai.prompts import configure as configure_prompts


def _get_version() -> str:
    """Read package version from installed metadata, with dev fallback."""
    try:
        return importlib.metadata.version("scout-ai")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0-dev"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup/shutdown lifecycle."""
    settings = AppSettings()
    validate_settings(settings)
    setup_logging(settings.observability)
    setup_tracing(settings.observability, tenant_id=settings.tenant_id, lob=settings.lob)

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

    app.state.settings = settings
    yield


_api_config = APIConfig()

app = FastAPI(
    title=_api_config.title,
    description=_api_config.description,
    version=_get_version(),
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(index.router, prefix="/api", dependencies=[Depends(require_auth)])
app.include_router(retrieve.router, prefix="/api", dependencies=[Depends(require_auth)])
app.include_router(extract.router, prefix="/api", dependencies=[Depends(require_auth)])
app.include_router(index_async.router, prefix="/api", dependencies=[Depends(require_auth)])
app.include_router(feedback.router, prefix="/api", dependencies=[Depends(require_auth)])
