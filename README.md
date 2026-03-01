# scout-ai

Standalone vectorless RAG module using internalized PageIndex tree indexing. Builds hierarchical tree indexes from pre-OCR'd document pages and uses LLM reasoning for retrieval and extraction — no vector embeddings required.

Designed for processing Attending Physician Statements (APS) and medical records on-premise with any OpenAI-compatible LLM endpoint (vLLM, LiteLLM, Ollama, or OpenAI).

## Features

- **Vectorless retrieval** — Tree-indexed search using LLM reasoning over document structure instead of embedding similarity
- **Any LLM backend** — Works with vLLM, LiteLLM, Ollama, or OpenAI via configurable `base_url`
- **Pre-OCR'd input** — Accepts `List[PageContent]` (page number + text); no PDF parsing dependencies
- **Medical-domain heuristics** — Regex-based section detection for 15 APS section types before falling back to LLM
- **Category-batched retrieval** — Groups questions by 16 extraction categories, reducing hundreds of LLM calls to ~16
- **Tiered extraction** — Tier 1 batches simple lookups (20 per prompt), Tier 2/3 uses individual reasoning chains
- **Plugin-ready interfaces** — Abstract base classes allow swapping retrieval/indexing providers

## Installation

```bash
pip install -e .

# With tiktoken tokenizer support:
pip install -e ".[tiktoken]"

# Development dependencies:
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Quick Start

### 1. Build a tree index

```bash
scout-ai index pages.json \
  --doc-id aps-001 \
  --doc-name "John Doe APS" \
  --base-url http://localhost:4000/v1 \
  --api-key your-key \
  --model qwen3-14b \
  --output index.json
```

Where `pages.json` is a JSON array of `{"page_number": int, "text": str}` objects.

### 2. Search the index

```bash
scout-ai retrieve index.json "blood pressure readings" \
  --base-url http://localhost:4000/v1 \
  --api-key your-key \
  --model qwen3-14b
```

### 3. Extract answers

```bash
scout-ai extract index.json questions.json \
  --base-url http://localhost:4000/v1 \
  --api-key your-key \
  --model qwen3-14b \
  --output results.json
```

Where `questions.json` is a JSON array of:
```json
[
  {
    "question_id": "demo-001",
    "category": "demographics",
    "question_text": "What is the patient's full name?",
    "tier": 1,
    "expected_type": "text"
  }
]
```

## Python API

```python
import asyncio
from scout_ai import (
    PageIndexSettings,
    PageContent,
    LLMClient,
    PageIndexIndexer,
    PageIndexRetrieval,
    PageIndexChat,
    ExtractionService,
    IngestionService,
    IndexStore,
    ExtractionQuestion,
)

settings = PageIndexSettings(
    llm_base_url="http://localhost:4000/v1",
    llm_api_key="your-key",
    llm_model="qwen3-14b",
)

client = LLMClient(settings)
indexer = PageIndexIndexer(settings, client)
retrieval = PageIndexRetrieval(settings, client)
chat = PageIndexChat(settings, client)

# Build index from pre-OCR'd pages
pages = [
    PageContent(page_number=1, text="FACE SHEET\nPatient Name: John Doe..."),
    PageContent(page_number=2, text="HISTORY AND PHYSICAL\n..."),
]

async def run():
    index = await indexer.build_index(pages, "doc-001", "John Doe APS")

    # Single query retrieval
    result = await retrieval.retrieve(index, "What medications is the patient taking?")
    print(result.source_pages, result.reasoning)

    # Batch extraction
    questions = [
        ExtractionQuestion(
            question_id="med-001",
            category="current_medications",
            question_text="List all current medications with dosages",
            tier=1,
            expected_type="list",
        ),
    ]
    service = ExtractionService(retrieval, chat)
    results = await service.extract(index, questions)
    for batch in results:
        for ext in batch.extractions:
            print(f"{ext.question_id}: {ext.answer} (confidence: {ext.confidence})")

asyncio.run(run())
```

## Configuration

All settings are configurable via environment variables with the `PAGEINDEX_` prefix:

| Variable | Default | Description |
|---|---|---|
| `PAGEINDEX_LLM_BASE_URL` | `http://localhost:4000/v1` | LLM API endpoint |
| `PAGEINDEX_LLM_API_KEY` | `no-key-required` | API key |
| `PAGEINDEX_LLM_MODEL` | `qwen3-14b` | Model name |
| `PAGEINDEX_LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `PAGEINDEX_LLM_TIMEOUT` | `120` | Request timeout (seconds) |
| `PAGEINDEX_TOC_CHECK_PAGE_COUNT` | `3` | Pages to scan for table of contents |
| `PAGEINDEX_MAX_PAGES_PER_NODE` | `4` | Max pages per tree node |
| `PAGEINDEX_MAX_TOKENS_PER_NODE` | `2000` | Max tokens per tree node |
| `PAGEINDEX_ENABLE_MEDICAL_CLASSIFICATION` | `true` | Enable APS section detection |
| `PAGEINDEX_RETRIEVAL_TOP_K_NODES` | `5` | Default nodes to retrieve |
| `PAGEINDEX_TOKENIZER_METHOD` | `approximate` | Token counting: `approximate`, `tiktoken`, or `transformers` |

## Architecture

```
src/scout_ai/
  config.py                           # Pydantic Settings (env-driven)
  models.py                           # All Pydantic data models
  exceptions.py                       # Exception hierarchy
  interfaces/                         # Abstract base classes
    ingestion.py                      #   IIngestionProvider
    retrieval.py                      #   IRetrievalProvider
    chat.py                           #   IChatProvider
  providers/pageindex/                # Internalized PageIndex implementation
    client.py                         #   AsyncOpenAI LLM client
    tokenizer.py                      #   Pluggable token counter
    indexer.py                        #   Tree index builder (3 modes)
    medical_classifier.py             #   APS section detection (regex + LLM)
    tree_builder.py                   #   Flat-list to tree conversion
    tree_utils.py                     #   Node traversal, mapping, serialization
    retrieval.py                      #   Single-query tree search
    batch_retrieval.py                #   Category-batched multi-question retrieval
    chat.py                           #   Tiered extraction completions
  services/
    ingestion_service.py              #   Index + persist
    extraction_service.py             #   Retrieve + extract pipeline
    index_store.py                    #   JSON file persistence
  aps/                                # APS medical domain
    categories.py                     #   16 extraction categories
    section_patterns.py               #   Medical section regex patterns
    prompts.py                        #   APS-specific prompt templates
  cli/
    main.py                           #   index / retrieve / extract commands
```

### Indexing Modes

The indexer supports three modes (matching vanilla PageIndex behavior):

1. **Mode 1 — TOC with page numbers**: Heuristic section detection finds boundaries via regex, then parses page ranges directly
2. **Mode 2 — TOC without page numbers**: Sections detected but no page numbers; LLM maps sections to pages
3. **Mode 3 — No TOC**: LLM generates document structure from content chunks

Medical heuristic pre-pass runs before any LLM-based detection, saving 10-30 LLM calls for typical APS documents.

## Testing

```bash
# Unit tests (no LLM required)
pytest tests/unit/ -v

# Integration tests (mocked LLM via respx)
pytest tests/integration/ -v

# All tests
pytest tests/ -v
```

## License

MIT
