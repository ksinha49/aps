"""FastAPI application with lifespan management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from scout_ai.api.routes import extract, health, index, retrieve
from scout_ai.core.config import AppSettings
from scout_ai.hooks import setup_logging, setup_tracing
from scout_ai.prompts import configure as configure_prompts


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup/shutdown lifecycle."""
    settings = AppSettings()
    setup_logging(settings.observability)
    setup_tracing(settings.observability)

    configure_prompts(
        backend=settings.prompt.backend,
        table_name=settings.prompt.table_name,
        aws_region=settings.prompt.aws_region,
        cache_ttl_seconds=settings.prompt.cache_ttl_seconds,
        cache_max_size=settings.prompt.cache_max_size,
        fallback_to_file=settings.prompt.fallback_to_file,
        default_lob=settings.prompt.default_lob,
        default_department=settings.prompt.default_department,
        default_use_case=settings.prompt.default_use_case,
        default_process=settings.prompt.default_process,
    )

    app.state.settings = settings
    yield


app = FastAPI(
    title="Scout AI by Ameritas",
    description="Vectorless RAG system with hierarchical tree indexes",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(index.router, prefix="/api")
app.include_router(retrieve.router, prefix="/api")
app.include_router(extract.router, prefix="/api")
