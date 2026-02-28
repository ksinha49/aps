"""CLI for pageindex-rag: index / retrieve / extract commands."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from pageindex_rag.config import PageIndexSettings
from pageindex_rag.models import (
    DocumentIndex,
    ExtractionQuestion,
    PageContent,
)
from pageindex_rag.providers.pageindex.client import LLMClient
from pageindex_rag.providers.pageindex.chat import PageIndexChat
from pageindex_rag.providers.pageindex.indexer import PageIndexIndexer
from pageindex_rag.providers.pageindex.retrieval import PageIndexRetrieval
from pageindex_rag.providers.pageindex.tree_utils import tree_to_toc_string
from pageindex_rag.services.extraction_service import ExtractionService
from pageindex_rag.services.index_store import IndexStore
from pageindex_rag.services.ingestion_service import IngestionService

app = typer.Typer(name="pageindex-rag", help="Vectorless RAG with tree-indexed retrieval")
console = Console()


def _build_settings(
    base_url: Optional[str],
    api_key: Optional[str],
    model: Optional[str],
) -> PageIndexSettings:
    """Build settings, overriding env defaults with CLI flags."""
    overrides: dict = {}
    if base_url:
        overrides["llm_base_url"] = base_url
    if api_key:
        overrides["llm_api_key"] = api_key
    if model:
        overrides["llm_model"] = model
    return PageIndexSettings(**overrides)


def _load_pages(pages_path: Path) -> list[PageContent]:
    """Load pages from a JSON file."""
    raw = json.loads(pages_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [PageContent(**item) for item in raw]
    raise typer.BadParameter(f"Expected JSON array in {pages_path}")


@app.command()
def index(
    pages_file: Path = typer.Argument(..., help="JSON file with pages"),
    doc_id: str = typer.Option(..., help="Unique document identifier"),
    doc_name: str = typer.Option("", help="Human-readable document name"),
    output: Optional[Path] = typer.Option(None, help="Output path for index JSON"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="LLM base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="LLM API key"),
    model: Optional[str] = typer.Option(None, "--model", help="LLM model name"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build a tree index from pages."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    settings = _build_settings(base_url, api_key, model)
    client = LLMClient(settings)
    indexer_impl = PageIndexIndexer(settings, client)

    if output:
        store = IndexStore(output.parent)
    else:
        store = IndexStore(settings.index_store_path)

    service = IngestionService(indexer_impl, store)

    console.print(f"[bold]Loading pages from {pages_file}[/bold]")
    pages = _load_pages(pages_file)
    console.print(f"Loaded {len(pages)} pages")

    async def _run() -> DocumentIndex:
        return await service.ingest(pages, doc_id, doc_name or doc_id, force=True)

    result = asyncio.run(_run())

    if output:
        output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]Index saved to {output}[/green]")
    else:
        idx_path = settings.index_store_path / f"{doc_id}.json"
        console.print(f"[green]Index saved to {idx_path}[/green]")

    console.print("\n[bold]Document Structure:[/bold]")
    console.print(tree_to_toc_string(result.tree))
    console.print(f"\nTotal nodes: {len(result.tree)}, Pages: {result.total_pages}")


@app.command()
def retrieve(
    index_file: Path = typer.Argument(..., help="Path to index JSON file"),
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, help="Number of nodes to retrieve"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    model: Optional[str] = typer.Option(None, "--model"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Search a tree index for relevant sections."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    settings = _build_settings(base_url, api_key, model)
    client = LLMClient(settings)
    retrieval_impl = PageIndexRetrieval(settings, client)

    console.print(f"[bold]Loading index from {index_file}[/bold]")
    doc_index = DocumentIndex.model_validate_json(
        index_file.read_text(encoding="utf-8")
    )

    async def _run():
        return await retrieval_impl.retrieve(doc_index, query, top_k=top_k)

    result = asyncio.run(_run())

    console.print(f"\n[bold]Query:[/bold] {result.query}")
    console.print(f"[bold]Reasoning:[/bold] {result.reasoning}")
    console.print(f"[bold]Source pages:[/bold] {result.source_pages}\n")

    table = Table(title="Retrieved Nodes")
    table.add_column("Node ID", style="cyan")
    table.add_column("Title", style="green")
    table.add_column("Pages")
    table.add_column("Text Preview", max_width=60)

    for node in result.retrieved_nodes:
        text_preview = node.get("text", "")
        if len(text_preview) > 100:
            text_preview = text_preview[:100] + "..."
        table.add_row(
            node["node_id"],
            node["title"],
            f"{node['start_index']}-{node['end_index']}",
            text_preview,
        )

    console.print(table)


@app.command()
def extract(
    index_file: Path = typer.Argument(..., help="Path to index JSON file"),
    questions_file: Path = typer.Argument(..., help="JSON file with extraction questions"),
    output: Optional[Path] = typer.Option(None, help="Output path for results JSON"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    model: Optional[str] = typer.Option(None, "--model"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Extract answers from an indexed document."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    settings = _build_settings(base_url, api_key, model)
    client = LLMClient(settings)
    retrieval_impl = PageIndexRetrieval(settings, client)
    chat_impl = PageIndexChat(settings, client)
    service = ExtractionService(retrieval_impl, chat_impl)

    console.print(f"[bold]Loading index from {index_file}[/bold]")
    doc_index = DocumentIndex.model_validate_json(
        index_file.read_text(encoding="utf-8")
    )

    console.print(f"[bold]Loading questions from {questions_file}[/bold]")
    raw_questions = json.loads(questions_file.read_text(encoding="utf-8"))
    questions = [ExtractionQuestion(**q) for q in raw_questions]
    console.print(f"Loaded {len(questions)} questions")

    async def _run():
        return await service.extract(doc_index, questions)

    results = asyncio.run(_run())

    output_data = [r.model_dump() for r in results]

    if output:
        output.write_text(
            json.dumps(output_data, indent=2, default=str), encoding="utf-8"
        )
        console.print(f"[green]Results saved to {output}[/green]")
    else:
        console.print(json.dumps(output_data, indent=2, default=str))

    total_answers = sum(len(r.extractions) for r in results)
    found = sum(
        1 for r in results for e in r.extractions if e.answer != "Not found"
    )
    console.print(
        f"\n[bold]Extracted {found}/{total_answers} answers "
        f"across {len(results)} categories[/bold]"
    )


if __name__ == "__main__":
    app()
