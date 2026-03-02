"""Environment-driven configuration via Pydantic Settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class ScoutSettings(BaseSettings):
    """All configuration is driven by env vars with ``SCOUT_`` prefix.

    Example::

        export SCOUT_LLM_BASE_URL=http://localhost:4000/v1
        export SCOUT_LLM_MODEL=qwen3-14b
    """

    model_config = {"env_prefix": "SCOUT_"}

    # ── LLM backend ─────────────────────────────────────────────────
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "no-key"
    llm_model: str = "qwen3-14b"
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_seed: int | None = None
    llm_timeout: float = 120.0
    llm_max_retries: int = 5
    retry_jitter_factor: float = 0.5
    retry_max_delay: float = 30.0

    # ── Indexing ─────────────────────────────────────────────────────
    toc_check_page_count: int = 20
    max_pages_per_node: int = 10
    max_tokens_per_node: int = 20_000
    max_recursion_depth: int = 10
    max_group_tokens: int = 20_000
    summary_max_chars: int = 4000
    classification_max_chars: int = 500
    toc_continuation_attempts: int = 3
    min_heuristic_sections: int = 3

    # ── Node enrichment ──────────────────────────────────────────────
    enable_node_summaries: bool = True
    enable_section_classification: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "enable_section_classification",
            "enable_medical_classification",
            "SCOUT_ENABLE_MEDICAL_CLASSIFICATION",
        ),
    )
    enable_doc_description: bool = False

    # ── Retrieval ────────────────────────────────────────────────────
    retrieval_max_concurrent: int = 8
    retrieval_top_k_nodes: int = 5

    # ── Persistence ──────────────────────────────────────────────────
    index_store_path: Path = Path("./indexes")

    # ── Tokenizer ────────────────────────────────────────────────────
    tokenizer_method: Literal["approximate", "tiktoken", "transformers"] = "approximate"
    tokenizer_model: str = "gpt-4o"
