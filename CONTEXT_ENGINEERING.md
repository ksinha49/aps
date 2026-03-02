# Context Engineering Modules

> **Audience**: Developers working on Scout AI who need to understand the compression, factoring, prefix stabilization, and caching modules in `src/scout_ai/context/`.

---

## Overview

The context engineering layer sits between retrieval (which finds relevant document nodes) and extraction (which sends context + questions to the LLM). It applies four independent optimizations to reduce LLM cost, improve latency, and eliminate redundant calls — all configurable via environment variables and disabled by default.

```
Retrieved Nodes
    │
    ▼
┌──────────────────┐
│ Prefix Stabilizer │  ← Deterministic ordering → maximizes cache hits
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│ Context Builder   │  ← build_cited_context() — unchanged
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│ Compressor        │  ← Replaces [:8000] hard truncation
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│ Layer Builder     │  ← Multi-breakpoint prompt cache hierarchy
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│ Result Cache      │  ← Skip LLM entirely on cache hit
└────────┬─────────┘
         │
    ▼
LLM Extraction Call
```

### Design Principles

1. **All disabled by default** — zero behavioral change until opted in via `SCOUT_*` env vars.
2. **Independent modules** — enable any combination; they compose but don't depend on each other.
3. **Follows existing patterns** — `@runtime_checkable` Protocol, pydantic-settings config, factory functions, lazy imports for optional deps.
4. **Wraps, doesn't replace** — each module wraps existing logic at the boundary rather than rewriting core extraction.

---

## Package Structure

```
src/scout_ai/context/
├── __init__.py                  # Public API: create_compressor(), create_context_cache()
├── protocols.py                 # IContextCompressor, IContextCache (runtime_checkable)
├── models.py                    # CompressedContext, ContextLayer, CacheEntry (dataclasses)
│
├── compression/                 # Statistical context compression
│   ├── __init__.py              # create_compressor() factory
│   ├── noop.py                  # NoOpCompressor — passthrough (default)
│   ├── entropic.py              # EntropicCompressor — sentence entropy filtering (no deps)
│   └── llmlingua.py             # LLMLinguaCompressor — token-level (optional dep)
│
├── factoring/                   # Multi-breakpoint prompt cache hierarchy
│   ├── __init__.py              # Re-exports
│   ├── breakpoint_strategy.py   # Assigns cache_control markers to layer boundaries
│   └── layer_builder.py         # ContextLayerBuilder — builds Anthropic-compatible messages
│
├── prefix/                      # Deterministic context ordering
│   ├── __init__.py              # Re-exports
│   ├── sort_strategies.py       # page_number, section_path, doc_id_page
│   └── stabilizer.py            # PrefixStabilizer — dispatches to sort strategy
│
└── cache/                       # Extraction result caching
    ├── __init__.py              # create_context_cache() factory
    ├── key_strategy.py          # compute_cache_key(), compute_index_hash()
    ├── memory.py                # MemoryCache — OrderedDict LRU with asyncio.Lock
    ├── s3.py                    # S3Cache — wraps IPersistenceBackend
    └── redis.py                 # RedisCache — redis.asyncio (optional dep)
```

---

## Module Reference

### 1. Prefix Stabilization

**Problem**: Retrieved nodes can arrive in different orders across calls for the same document + question combination. Different ordering = different prompt string = prompt cache miss.

**Solution**: Sort nodes deterministically before building the context string, so identical retrieval results always produce identical context.

**Configuration**:

```bash
export SCOUT_PREFIX_ENABLED=true
export SCOUT_PREFIX_SORT_STRATEGY=page_number   # page_number | section_path | doc_id_page
export SCOUT_PREFIX_DETERMINISTIC_JSON=true
```

**Sort strategies**:

| Strategy | Sort Key | Best For |
|----------|----------|----------|
| `page_number` | `(start_index, node_id)` | Single-document workflows (default) |
| `section_path` | `(section_path, start_index)` | Tree-order consistency |
| `doc_id_page` | `(doc_id, start_index)` | Multi-document pipelines |

**Usage** (automatic when enabled, or manual):

```python
from scout_ai.context.prefix import PrefixStabilizer

stabilizer = PrefixStabilizer("page_number")
sorted_nodes = stabilizer.stabilize(retrieved_nodes)

# Also: deterministic JSON serialization
json_str = PrefixStabilizer.stabilize_json({"b": 2, "a": 1})
# '{"a":1,"b":2}'
```

**Integration points**:
- `agents/orchestrator.py` — applied to `retrieval.retrieved_nodes` before `build_cited_context()`
- `services/extraction_service.py` — same for legacy path

---

### 2. Context Compression

**Problem**: The legacy path hard-truncates context at `[:8000]` characters, discarding potentially important information at the end. Long boilerplate text wastes tokens.

**Solution**: Replace hard truncation with intelligent compression that preserves high-information sentences and drops boilerplate.

**Configuration**:

```bash
export SCOUT_COMPRESSION_ENABLED=true
export SCOUT_COMPRESSION_METHOD=entropic     # noop | entropic | llmlingua
export SCOUT_COMPRESSION_TARGET_RATIO=0.5    # 0.0–1.0 (lower = more aggressive)
export SCOUT_COMPRESSION_MIN_TOKENS_FOR_COMPRESSION=500
```

**Compressor implementations**:

| Backend | Deps | Approach | Typical Ratio |
|---------|------|----------|---------------|
| `noop` | None | Passthrough | 1.0 |
| `entropic` | None | Sentence-level entropy filtering via word frequency | 0.3–0.7 |
| `llmlingua` | `pip install scout-ai[llmlingua]` | Token-level pruning with LLMLingua-2 | 0.2–0.5 |

**Protocol**:

```python
from scout_ai.context.protocols import IContextCompressor
from scout_ai.context.models import CompressedContext

class IContextCompressor(Protocol):
    def compress(self, text: str, *, target_ratio: float = 0.5) -> CompressedContext: ...
```

**How `EntropicCompressor` works**:
1. Split text into sentences
2. Compute word frequencies across the entire document
3. Score each sentence by information entropy (rare words = high entropy)
4. Sort sentences by entropy (highest first)
5. Keep top sentences until `target_ratio` of original length is reached
6. Re-order kept sentences by their original position

**Integration points**:
- `agents/orchestrator.py` — compresses context after `build_cited_context()`, before extraction
- `services/extraction_service.py` — same for legacy path
- `providers/pageindex/chat.py` — `_prepare_context()` replaces `context[:8000]` in both `_extract_batch()` and `_extract_individual()`

---

### 3. Context Factoring (Multi-Breakpoint Caching)

**Problem**: The existing prompt caching puts one `cache_control` breakpoint on the system message. Anthropic supports up to 4 breakpoints — unused breakpoints are wasted cache savings.

**Solution**: Factor the prompt into layers (system → tools → document → query) with cache breakpoints at layer boundaries, maximizing cache reuse.

**Configuration**:

```bash
# Uses existing SCOUT_CACHING_ prefix
export SCOUT_CACHING_ENABLED=true
export SCOUT_CACHING_MAX_BREAKPOINTS=4          # 1–4 (Anthropic limit)
export SCOUT_CACHING_CACHE_DOCUMENT_LAYER=true
export SCOUT_CACHING_CACHE_TOOL_LAYER=false
```

**Layer hierarchy** (most stable → least stable):

```
system prompt     →  [cache_control]  rarely changes; shared across all calls
tool definitions  →  [cache_control]  optional; stable across calls
document context  →  [cache_control]  changes per document; shared across questions
user query        →  (no cache)       changes every call
```

**Breakpoint priority**: When `max_breakpoints` is less than the number of cacheable layers, the system assigns breakpoints by priority: system > tools > document. The query layer is never cached.

**Usage**:

```python
from scout_ai.context.factoring import ContextLayerBuilder

builder = ContextLayerBuilder(max_breakpoints=4)

# Returns Anthropic/OpenAI-compatible messages with cache_control
messages = builder.build_messages(
    system_prompt="You are an extraction assistant.",
    document_context="Patient has diabetes...",
    query="What is the HbA1c level?",
)
# messages[0]["content"] = [
#   {"type": "text", "text": "You are...", "cache_control": {"type": "ephemeral"}},
#   {"type": "text", "text": "Patient has...", "cache_control": {"type": "ephemeral"}},
# ]
# messages[1] = {"role": "user", "content": "What is the HbA1c level?"}
```

**Integration points**:
- `providers/pageindex/client.py` — `_build_layered_messages()` method wraps `ContextLayerBuilder`

---

### 4. Extraction Result Cache

**Problem**: The same question asked against the same document with the same model produces the same answer — but we call the LLM every time.

**Solution**: Cache extraction results keyed by `(question_id, index_hash, model_name)`. On cache hit, skip the LLM entirely.

**Configuration**:

```bash
export SCOUT_CONTEXT_CACHE_ENABLED=true
export SCOUT_CONTEXT_CACHE_BACKEND=memory      # memory | s3 | redis
export SCOUT_CONTEXT_CACHE_TTL_SECONDS=3600
export SCOUT_CONTEXT_CACHE_MAX_ENTRIES=1000
export SCOUT_CONTEXT_CACHE_L1_MAX_SIZE=100
export SCOUT_CONTEXT_CACHE_REDIS_URL=redis://localhost:6379
```

**Cache backends**:

| Backend | Persistence | Best For |
|---------|-------------|----------|
| `memory` | Process-local (lost on restart) | Development, single-instance |
| `s3` | Durable (wraps `IPersistenceBackend`) | Multi-instance, shared cache |
| `redis` | Durable (`pip install scout-ai[redis]`) | High-throughput, low-latency |

**Cache key computation**:

```python
from scout_ai.context.cache.key_strategy import compute_cache_key, compute_index_hash

# Stable hash from document index metadata
index_hash = compute_index_hash(document_index)  # 16-char hex

# Deterministic SHA-256 key
key = compute_cache_key(
    question_id="q1",
    index_hash=index_hash,
    model_name="claude-3-5-sonnet",
    context_hash="",  # optional
)
```

**Protocol**:

```python
from scout_ai.context.protocols import IContextCache

class IContextCache(Protocol):
    async def get(self, key: str) -> Any | None: ...
    async def put(self, key: str, value: Any, *, ttl_seconds: int = 0) -> None: ...
    async def invalidate(self, key: str) -> None: ...
    async def clear(self) -> None: ...
```

**MemoryCache internals**:
- `OrderedDict` for LRU ordering — `move_to_end()` on access, `popitem(last=False)` for eviction
- `asyncio.Lock` for safe concurrent coroutine access
- TTL checked on `get()` — expired entries are lazily removed

---

## Integration Architecture

### Where Modules Hook In

```
ExtractionPipeline.run()
│
├── retrieval = await self._retrieval.batch_retrieve(index, questions)
│
├── for each category:
│   │
│   ├── nodes = retrieval.retrieved_nodes
│   │
│   ├── if prefix_stabilizer:                         ◄─── PREFIX
│   │       nodes = prefix_stabilizer.stabilize(nodes)
│   │
│   ├── context = build_cited_context(nodes, page_map)
│   │
│   ├── if compressor:                                ◄─── COMPRESSION
│   │       context = compressor.compress(context).text
│   │
│   └── extractions = await chat.extract_answers(questions, context)
│       │
│       └── ScoutChat._prepare_context(context)       ◄─── COMPRESSION (also in chat)
│           └── if self._compressor:
│                   return compressor.compress(context).text
│               else:
│                   return context[:self._max_context_chars]
│
└── return results
```

### Wiring in `create_extraction_pipeline()`

The factory in `agents/orchestrator.py` reads `AppSettings` and wires modules when enabled:

```python
if settings.prefix.enabled:
    prefix_stabilizer = PrefixStabilizer(strategy=settings.prefix.sort_strategy)

if settings.compression.enabled:
    compressor = create_compressor(settings)

if settings.context_cache.enabled:
    context_cache = create_context_cache(settings)

return ExtractionPipeline(
    retrieval, chat,
    prefix_stabilizer=prefix_stabilizer,
    compressor=compressor,
    context_cache=context_cache,
)
```

---

## Configuration Reference

| Env Var | Type | Default | Description |
|---------|------|---------|-------------|
| `SCOUT_COMPRESSION_ENABLED` | bool | `false` | Enable context compression |
| `SCOUT_COMPRESSION_METHOD` | str | `noop` | `noop`, `entropic`, `llmlingua` |
| `SCOUT_COMPRESSION_TARGET_RATIO` | float | `0.5` | Target compressed/original ratio |
| `SCOUT_COMPRESSION_MIN_TOKENS_FOR_COMPRESSION` | int | `500` | Skip compression for short text |
| `SCOUT_PREFIX_ENABLED` | bool | `false` | Enable prefix stabilization |
| `SCOUT_PREFIX_SORT_STRATEGY` | str | `page_number` | `page_number`, `section_path`, `doc_id_page` |
| `SCOUT_PREFIX_DETERMINISTIC_JSON` | bool | `true` | Sort keys in JSON serialization |
| `SCOUT_CONTEXT_CACHE_ENABLED` | bool | `false` | Enable extraction result caching |
| `SCOUT_CONTEXT_CACHE_BACKEND` | str | `memory` | `memory`, `s3`, `redis` |
| `SCOUT_CONTEXT_CACHE_TTL_SECONDS` | int | `3600` | Cache entry time-to-live |
| `SCOUT_CONTEXT_CACHE_MAX_ENTRIES` | int | `1000` | Max entries (LRU eviction) |
| `SCOUT_CONTEXT_CACHE_L1_MAX_SIZE` | int | `100` | L1 in-memory size |
| `SCOUT_CONTEXT_CACHE_REDIS_URL` | str | `""` | Redis connection URL |
| `SCOUT_CACHING_MAX_BREAKPOINTS` | int | `4` | Max cache breakpoints (1–4) |
| `SCOUT_CACHING_CACHE_DOCUMENT_LAYER` | bool | `true` | Cache document context layer |
| `SCOUT_CACHING_CACHE_TOOL_LAYER` | bool | `false` | Cache tool definitions layer |

---

## Testing

### Test Files

```
tests/unit/
├── test_context_compression.py     # 13 tests: NoOp, Entropic, LLMLingua import error, factory
├── test_context_prefix.py          # 10 tests: sort strategies, stabilizer, deterministic JSON
├── test_context_factoring.py       # 12 tests: breakpoint placement, layer builder, message format
├── test_context_cache.py           # 14 tests: MemoryCache LRU, TTL, FakeContextCache
├── test_context_cache_key.py       #  8 tests: key determinism, index hash stability
├── test_context_config.py          #  8 tests: config defaults, env overrides

tests/integration/
├── test_compressed_extraction.py   #  3 tests: compressor + ScoutChat integration
├── test_factored_extraction.py     #  4 tests: layered messages + cache_control
├── test_context_cache_pipeline.py  #  6 tests: cache + extraction pipeline

tests/fakes/
└── fake_context_cache.py           # Dict-backed IContextCache for tests
```

### Running Tests

```bash
# All context tests
pytest tests/unit/test_context_*.py tests/integration/test_compressed_extraction.py \
       tests/integration/test_factored_extraction.py tests/integration/test_context_cache_pipeline.py -v

# With specific env vars enabled
SCOUT_COMPRESSION_ENABLED=true SCOUT_COMPRESSION_METHOD=entropic \
    pytest tests/integration/test_compressed_extraction.py -v

SCOUT_PREFIX_ENABLED=true \
    pytest tests/integration/test_factored_extraction.py -v

SCOUT_CONTEXT_CACHE_ENABLED=true \
    pytest tests/integration/test_context_cache_pipeline.py -v
```

---

## Extending

### Adding a New Compressor

1. Create `src/scout_ai/context/compression/my_compressor.py`:

```python
from scout_ai.context.models import CompressedContext

class MyCompressor:
    def compress(self, text: str, *, target_ratio: float = 0.5) -> CompressedContext:
        compressed = my_compression_logic(text, target_ratio)
        return CompressedContext(
            text=compressed,
            original_length=len(text),
            compressed_length=len(compressed),
            compression_ratio=len(compressed) / len(text),
            method="my_method",
        )
```

2. Register in `compression/__init__.py` factory:

```python
elif method == "my_method":
    from scout_ai.context.compression.my_compressor import MyCompressor
    return MyCompressor()
```

3. Update `CompressionConfig.method` Literal type to include `"my_method"`.

### Adding a New Cache Backend

1. Implement the `IContextCache` protocol (4 async methods: `get`, `put`, `invalidate`, `clear`).
2. Register in `cache/__init__.py` factory.
3. Update `ContextCacheConfig.backend` Literal type.

### Adding a New Sort Strategy

1. Add a function to `prefix/sort_strategies.py` with signature `(list[dict]) -> list[dict]`.
2. Register in `prefix/stabilizer.py` `_STRATEGIES` dict.
3. Update `PrefixConfig.sort_strategy` Literal type.
