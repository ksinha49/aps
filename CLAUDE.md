# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development

```bash
# Install (editable, with dev tools)
pip install -e ".[dev]"

# Install with optional extras
pip install -e ".[dev,api,otel,s3]"

# Run all tests (no LLM required — integration tests use respx mocks)
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only (mocked LLM via respx)
pytest tests/integration/ -v

# Single test file
pytest tests/unit/test_tree_builder.py -v

# Single test
pytest tests/unit/test_models.py::TestPageContent::test_basic -v

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Run API server locally
uvicorn pageindex_rag.api.app:app --host 0.0.0.0 --port 8080

# Docker (production)
docker compose -f docker/docker-compose.yml up --build

# Docker (development with hot reload)
docker compose -f docker/docker-compose.dev.yml up --build
```

All async tests use `asyncio_mode = "auto"` (configured in pyproject.toml), so `@pytest.mark.asyncio` is optional but sometimes used explicitly in integration tests.

## Architecture

Vectorless RAG system that builds hierarchical tree indexes from pre-OCR'd document pages and uses LLM reasoning for retrieval — no embeddings. Orchestrated by **Strands Agents SDK** with lifecycle hooks for observability, cost tracking, and resilience.

### Data Flow

```
PageContent[] → IndexingAgent (Strands) → DocumentIndex (tree of TreeNode[])
                  ├── detect_toc        (skill)
                  ├── process_toc       (skill, 3 modes)
                  ├── verify_toc        (skill)
                  ├── split_large_nodes (skill)
                  └── enrich_nodes      (skill)
                                                ↓
ExtractionQuestion[] → ExtractionPipeline → BatchExtractionResult[]
                        ├── RetrievalAgent → batch_retrieve (1 LLM call per category)
                        └── ExtractionAgent → extract_batch / extract_individual (tiered)
```

### Directory Structure

```
src/pageindex_rag/
├── __init__.py                   # Public API (legacy + Strands-era exports)
├── config.py                     # Legacy PageIndexSettings
├── models.py                     # All Pydantic data models
├── exceptions.py                 # Exception hierarchy
│
├── core/                         # Strands-era foundation
│   ├── config.py                 # Nested AppSettings (LLM, indexing, extraction, etc.)
│   ├── types.py                  # Type aliases (JsonDict, PageMap, etc.)
│   └── exceptions.py             # Extended exceptions (re-exports originals)
│
├── agents/                       # Strands Agent factories
│   ├── factory.py                # create_model() — provider switch (Bedrock/OpenAI/Ollama/LiteLLM)
│   ├── indexing_agent.py         # create_indexing_agent(settings)
│   ├── retrieval_agent.py        # create_retrieval_agent(settings)
│   ├── extraction_agent.py       # create_extraction_agent(settings)
│   └── orchestrator.py           # ExtractionPipeline — sequential retrieve→extract
│
├── skills/                       # @tool-decorated Strands skills
│   ├── common/                   # Shared: json_parser, token_counter
│   ├── indexing/                  # detect_toc, process_toc, verify_toc, split_nodes, enrich_nodes, build_index
│   ├── retrieval/                # tree_search, batch_retrieve
│   └── extraction/               # extract_batch, extract_individual
│
├── hooks/                        # Strands HookProvider implementations
│   ├── audit_hook.py             # Logs every LLM call + tool execution
│   ├── cost_hook.py              # Token usage accumulator (ContextVar)
│   ├── checkpoint_hook.py        # Pipeline state checkpointing for resume
│   ├── circuit_breaker_hook.py   # CLOSED→OPEN→HALF_OPEN failure protection
│   ├── dead_letter_hook.py       # Failed tool capture for later analysis
│   ├── tracing.py                # OpenTelemetry OTLP setup
│   └── logging_config.py         # structlog configuration
│
├── prompts/                      # Prompt registry + templates
│   ├── registry.py               # get_prompt(domain, category, name)
│   └── templates/
│       ├── base/                  # Agent system prompts (indexing, retrieval, extraction)
│       └── aps/                   # APS domain prompts (indexing, retrieval, extraction, classification)
│
├── domains/                      # Domain-specific modules
│   └── aps/                      # APS (Attending Physician Statement)
│       ├── models.py             # Re-exports ExtractionCategory, MedicalSectionType
│       ├── categories.py         # 16 category descriptions
│       ├── section_patterns.py   # Regex patterns for section detection
│       └── classifier.py         # MedicalSectionClassifier (regex-first, LLM fallback)
│
├── persistence/                  # Pluggable storage backends
│   ├── protocols.py              # IPersistenceBackend Protocol
│   ├── file_backend.py           # Local JSON file storage
│   ├── s3_backend.py             # AWS S3 storage
│   └── memory_backend.py         # Dict-backed (for tests)
│
├── providers/pageindex/          # Legacy provider implementations (still functional)
│   ├── client.py                 # LLMClient (AsyncOpenAI wrapper)
│   ├── indexer.py                # PageIndexIndexer (3-mode cascade)
│   ├── retrieval.py              # PageIndexRetrieval
│   ├── batch_retrieval.py        # BatchRetrieval (category-grouped)
│   ├── chat.py                   # PageIndexChat (tiered extraction)
│   ├── medical_classifier.py     # Re-export shim → domains.aps.classifier
│   ├── tree_builder.py           # Tree construction
│   ├── tree_utils.py             # Tree traversal utilities
│   └── tokenizer.py              # Token counter (3 backends)
│
├── services/                     # Legacy orchestration layer
│   ├── extraction_service.py     # Wires retrieval → chat
│   ├── ingestion_service.py      # Wires indexer → IndexStore
│   └── index_store.py            # JSON file persistence
│
├── aps/                          # Re-export shims → domains/aps/
│
├── api/                          # FastAPI HTTP layer
│   ├── app.py                    # Application with lifespan
│   ├── routes/                   # health, index, retrieve, extract
│   └── middleware/               # Error handler
│
└── cli/main.py                   # Typer CLI: index, retrieve, extract
```

### Key Patterns

- **Strands Agent architecture**: Each agent (indexing, retrieval, extraction) is created via a factory function that takes `AppSettings`, builds the appropriate Strands `Agent` with `@tool` skills and `HookProvider` hooks.
- **Skill decomposition**: Each `@tool` returns JSON instructions for the agent to reason over. Companion pure-logic functions (`*_sync`, `resolve_*`, `parse_*`) handle deterministic post-processing without LLM calls.
- **Hook lifecycle**: `AuditHook` logs all activity, `CostHook` tracks tokens via ContextVar, `CheckpointHook` persists state after tool success, `CircuitBreakerHook` prevents cascading failures, `DeadLetterHook` captures failures.
- **Model provider factory**: `agents/factory.py` switches between `BedrockModel`, `OpenAIModel`, `OllamaModel`, `LiteLLMModel` based on `settings.llm.provider`.
- **Legacy backward compat**: All original imports from `pageindex_rag.*` still work. Old `aps/` and `providers/pageindex/medical_classifier.py` files are thin re-export shims.
- **Prompt registry**: `get_prompt("aps", "indexing", "DETECT_TOC_PROMPT")` lazy-loads from `prompts/templates/aps/indexing.py`.
- **Pluggable persistence**: `IPersistenceBackend` Protocol with file, S3, and memory implementations.
- All LLM interaction in legacy path goes through `LLMClient.complete()` / `complete_batch()`.
- The indexer has a fallback cascade: heuristic → Mode 1 → Mode 2 → Mode 3, each progressively more LLM-dependent.
- Integration tests mock the OpenAI API at the HTTP level using `respx`.

### Configuration

All settings use pydantic-settings with env var prefixes:

| Prefix | Config Class | Purpose |
|--------|-------------|---------|
| `PAGEINDEX_LLM_` | `LLMConfig` | Model provider, API key, temperature |
| `PAGEINDEX_INDEXING_` | `IndexingConfig` | TOC detection, node limits |
| `PAGEINDEX_ENRICHMENT_` | `EnrichmentConfig` | Summary, classification toggles |
| `PAGEINDEX_RETRIEVAL_` | `RetrievalConfig` | Concurrency, top-k |
| `PAGEINDEX_EXTRACTION_` | `ExtractionConfig` | Context limits, batch size |
| `PAGEINDEX_PERSISTENCE_` | `PersistenceConfig` | Backend type, S3 bucket |
| `PAGEINDEX_TOKENIZER_` | `TokenizerConfig` | Counter method |
| `PAGEINDEX_OBSERVABILITY_` | `ObservabilityConfig` | Tracing, OTLP endpoint, log level |

### Deployment Targets

- **AWS ECS (Fargate)**: `deploy/ecs/` — task definition + service config
- **AWS EKS**: `deploy/eks/` — deployment, service, configmap, HPA
- **RHEL on-premise**: `deploy/rhel/` — systemd unit, install script, logrotate
- **Docker**: `docker/` — production (RHEL UBI 9) and dev Dockerfiles, compose files

### Test Fakes

- `tests/fakes/fake_model.py` — `FakeStrandsModel`: canned responses, call recording, Strands streaming events
- `tests/fakes/fake_persistence.py` — `FakePersistenceBackend`: dict-backed IPersistenceBackend

## Code Style

- **Line length**: 120 (ruff configured)
- **Target Python**: 3.10+
- **Ruff rules**: E, F, I, W
- **mypy**: strict mode enabled
- **Imports**: Use `from __future__ import annotations` in all modules
