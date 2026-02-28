"""Search endpoint for document tree index."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from pageindex_rag.api.routes._prompt_context import PromptContextOverride, apply_prompt_context

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    """Request to search a document index."""

    doc_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    prompt_context: PromptContextOverride | None = None


class SearchResponse(BaseModel):
    """Search results."""

    query: str
    source_pages: list[int]
    reasoning: str = ""


@router.post("/retrieve", response_model=SearchResponse)
async def search_index(request: SearchRequest, req: Request) -> SearchResponse:
    """Search the document tree index for sections matching a query."""
    settings = req.app.state.settings
    apply_prompt_context(request.prompt_context, settings.prompt)

    from pageindex_rag.services.index_store import IndexStore

    store = IndexStore(base_path=settings.persistence.store_path)
    index = store.load(request.doc_id)

    from pageindex_rag.providers.pageindex.retrieval import PageIndexRetrieval

    provider = PageIndexRetrieval(settings=settings)
    result = await provider.retrieve(
        query=request.query,
        index=index,
        top_k=request.top_k,
    )

    return SearchResponse(
        query=result.query,
        source_pages=result.source_pages,
        reasoning=result.reasoning,
    )
