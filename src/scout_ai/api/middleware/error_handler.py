"""Global exception handlers mapping domain exceptions to HTTP responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from scout_ai.exceptions import (
    ExtractionError,
    IndexBuildError,
    RetrievalError,
    ScoutError,
)


def register_error_handlers(app: FastAPI) -> None:
    """Register exception-to-HTTP-status mappings."""

    @app.exception_handler(IndexBuildError)
    async def handle_index_error(request: Request, exc: IndexBuildError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"error": str(exc), "type": "index_build_error"})

    @app.exception_handler(RetrievalError)
    async def handle_search_error(request: Request, exc: RetrievalError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"error": str(exc), "type": "search_error"})

    @app.exception_handler(ExtractionError)
    async def handle_extraction_error(request: Request, exc: ExtractionError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"error": str(exc), "type": "extraction_error"})

    @app.exception_handler(ScoutError)
    async def handle_generic_error(request: Request, exc: ScoutError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"error": str(exc), "type": "scout_error"})

    @app.exception_handler(KeyError)
    async def handle_not_found(request: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": str(exc), "type": "not_found"})
