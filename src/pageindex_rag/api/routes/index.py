"""Index creation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from pageindex_rag.api.routes._prompt_context import PromptContextOverride, apply_prompt_context

router = APIRouter(tags=["indexing"])


class IndexRequest(BaseModel):
    """Request to build a document index."""

    doc_id: str
    doc_name: str
    pages: list[dict[str, object]] = Field(
        ..., description="List of {page_number, text} objects"
    )
    prompt_context: PromptContextOverride | None = None


class IndexResponse(BaseModel):
    """Response from index creation."""

    doc_id: str
    total_pages: int
    node_count: int
    status: str = "created"


@router.post("/index", response_model=IndexResponse)
async def create_index(request: IndexRequest, req: Request) -> IndexResponse:
    """Build a hierarchical tree index from document pages.

    Delegates to the indexing agent via Strands.
    """
    settings = req.app.state.settings
    apply_prompt_context(request.prompt_context, settings.prompt)

    from pageindex_rag.models import PageContent
    from pageindex_rag.services.ingestion_service import IngestionService

    pages = [
        PageContent(page_number=p["page_number"], text=p["text"])  # type: ignore[arg-type]
        for p in request.pages
    ]

    service = IngestionService(settings=settings)
    index = await service.ingest(
        doc_id=request.doc_id,
        doc_name=request.doc_name,
        pages=pages,
    )

    return IndexResponse(
        doc_id=index.doc_id,
        total_pages=index.total_pages,
        node_count=len(index.tree),
    )
