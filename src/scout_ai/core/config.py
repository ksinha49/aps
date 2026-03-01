"""Nested pydantic-settings configuration for the application.

Supports both the new nested ``AppSettings().llm.provider`` style and
the flat ``SCOUT_LLM_MODEL`` env-var style via sub-model env prefixes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """LLM backend configuration.

    Env vars use ``SCOUT_LLM_`` prefix::

        export SCOUT_LLM_PROVIDER=ollama
        export SCOUT_LLM_MODEL=qwen3-14b
    """

    model_config = {"env_prefix": "SCOUT_LLM_"}

    provider: Literal["bedrock", "openai", "ollama", "litellm", "anthropic"] = "ollama"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "no-key"
    model: str = "qwen3-14b"
    temperature: float = 0.0
    timeout: float = 120.0
    max_retries: int = 5
    aws_region: str = "us-west-2"


class IndexingConfig(BaseSettings):
    """Indexing pipeline configuration.

    Env vars use ``SCOUT_INDEXING_`` prefix.
    """

    model_config = {"env_prefix": "SCOUT_INDEXING_"}

    toc_check_page_count: int = 20
    max_pages_per_node: int = 10
    max_tokens_per_node: int = 20_000
    max_recursion_depth: int = 10
    max_group_tokens: int = 20_000
    summary_max_chars: int = 4000


class EnrichmentConfig(BaseSettings):
    """Node enrichment toggles.

    Env vars use ``SCOUT_ENRICHMENT_`` prefix.
    """

    model_config = {"env_prefix": "SCOUT_ENRICHMENT_"}

    enable_node_summaries: bool = True
    enable_medical_classification: bool = True
    enable_doc_description: bool = False


class RetrievalConfig(BaseSettings):
    """Retrieval configuration.

    Env vars use ``SCOUT_RETRIEVAL_`` prefix.
    """

    model_config = {"env_prefix": "SCOUT_RETRIEVAL_"}

    max_concurrent: int = 8
    top_k_nodes: int = 5


class ExtractionConfig(BaseSettings):
    """Extraction pipeline configuration.

    Env vars use ``SCOUT_EXTRACTION_`` prefix.
    """

    model_config = {"env_prefix": "SCOUT_EXTRACTION_"}

    max_context_chars: int = 8000
    tier1_batch_size: int = 20


class PersistenceConfig(BaseSettings):
    """Persistence configuration.

    Env vars use ``SCOUT_PERSISTENCE_`` prefix.
    """

    model_config = {"env_prefix": "SCOUT_PERSISTENCE_"}

    backend: Literal["file", "s3", "memory"] = "file"
    store_path: Path = Path("./indexes")
    s3_bucket: str = ""
    s3_prefix: str = "indexes/"


class TokenizerConfig(BaseSettings):
    """Tokenizer configuration.

    Env vars use ``SCOUT_TOKENIZER_`` prefix.
    """

    model_config = {"env_prefix": "SCOUT_TOKENIZER_"}

    method: Literal["approximate", "tiktoken", "transformers"] = "approximate"
    model: str = "gpt-4o"


class PromptConfig(BaseSettings):
    """Prompt backend configuration.

    Env vars use ``SCOUT_PROMPT_`` prefix.
    """

    model_config = {"env_prefix": "SCOUT_PROMPT_"}

    backend: Literal["file", "dynamodb"] = "file"
    table_name: str = "scout-prompts"
    aws_region: str = "us-east-1"
    cache_ttl_seconds: float = 300.0
    cache_max_size: int = 500
    fallback_to_file: bool = True
    default_lob: str = "*"
    default_department: str = "*"
    default_use_case: str = "*"
    default_process: str = "*"


class ObservabilityConfig(BaseSettings):
    """Observability configuration.

    Env vars use ``SCOUT_OBSERVABILITY_`` prefix.
    """

    model_config = {"env_prefix": "SCOUT_OBSERVABILITY_"}

    enable_tracing: bool = False
    otlp_endpoint: str = "http://localhost:4317"
    service_name: str = "scout-ai"
    log_level: str = "INFO"


class CachingConfig(BaseSettings):
    """Prompt caching configuration for Anthropic/Bedrock providers.

    Env vars use ``SCOUT_CACHING_`` prefix::

        export SCOUT_CACHING_ENABLED=true
        export SCOUT_CACHING_MIN_CACHEABLE_TOKENS=1024
    """

    model_config = {"env_prefix": "SCOUT_CACHING_"}

    enabled: bool = False
    cache_system_prompt: bool = True
    cache_document_context: bool = True
    min_cacheable_tokens: int = 1024
    keepalive_interval_seconds: float = 240.0
    ttl_type: Literal["ephemeral", "long"] = "ephemeral"


class PDFFormattingConfig(BaseSettings):
    """PDF output formatting configuration.

    Env vars use ``SCOUT_PDF_`` prefix::

        export SCOUT_PDF_PAGE_SIZE=a4
        export SCOUT_PDF_INCLUDE_APPENDIX=false
    """

    model_config = {"env_prefix": "SCOUT_PDF_"}

    page_size: Literal["letter", "a4"] = "letter"
    margin_inches: float = Field(default=0.75, gt=0.0, le=3.0)
    font_family: str = "Helvetica"
    body_font_size: int = Field(default=10, ge=6, le=72)
    heading_font_size: int = Field(default=14, ge=6, le=72)
    include_appendix: bool = True
    include_cover_page: bool = True
    company_name: str = ""
    confidential_watermark: bool = True


class AppSettings(BaseSettings):
    """Top-level application settings aggregating all sub-configs.

    Each sub-config reads its own ``SCOUT_<GROUP>_*`` env vars.
    """

    llm: LLMConfig = LLMConfig()
    indexing: IndexingConfig = IndexingConfig()
    enrichment: EnrichmentConfig = EnrichmentConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    persistence: PersistenceConfig = PersistenceConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    prompt: PromptConfig = PromptConfig()
    caching: CachingConfig = CachingConfig()
    pdf: PDFFormattingConfig = PDFFormattingConfig()
