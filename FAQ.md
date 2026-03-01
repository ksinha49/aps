# FAQ: Pluggable Inference Backend Layer

> Questions organized by role. Each answer is self-contained — you don't need to read the full ARCHITECTURE.md first.

---

## For Developers

### Q: How do I switch inference backends?

Set one environment variable:

```bash
# Default — real-time (current behavior, no changes needed)
export SCOUT_LLM_INFERENCE_BACKEND=realtime

# Bedrock Batch (external package)
export SCOUT_LLM_INFERENCE_BACKEND=ameritas_bedrock_batch.backend:BedrockBatchBackend

# IDP adapter (external package)
export SCOUT_LLM_INFERENCE_BACKEND=ameritas_idp.adapter:IDPInferenceBackend
```

No code changes. Same Docker image. The factory resolves the class at startup.

---

### Q: How do I build a custom inference backend?

Implement two async methods in any Python class:

```python
class MyBackend:
    def __init__(self, settings):
        # settings is the full AppSettings object
        self._region = settings.llm.aws_region

    async def infer(self, messages, model, **params):
        # Call your inference service
        return InferenceResult(content="response text", finish_reason="finished")

    async def infer_batch(self, requests):
        # Process multiple requests
        results = []
        for req in requests:
            result = await self.infer(req.messages, req.model, **req.params)
            result.request_id = req.request_id
            results.append(result)
        return results
```

Your class does **not** need to inherit from anything or import Scout AI. It just needs to match the method signatures. This is Python's structural subtyping (duck typing via Protocol).

---

### Q: How do I test code that uses the inference backend?

Use the built-in `FakeInferenceBackend`:

```python
from tests.fakes.fake_inference import FakeInferenceBackend

backend = FakeInferenceBackend(default_content="mocked answer")
client = LLMClient(settings, backend=backend)

result = await client.complete("What is the diagnosis?")
assert result == "mocked answer"
assert len(backend.calls) == 1  # Verify the call was made
```

No `unittest.mock.patch` needed. No litellm mocking. No HTTP-level stubbing. The fake records every call for assertions.

---

### Q: Does retry logic still work with a pluggable backend?

Yes. Retry logic, error classification (`RetryableError` vs `NonRetryableError`), and message construction all remain in `LLMClient`. The backend is only responsible for the raw inference call. If your backend raises an exception, `LLMClient`'s existing retry loop handles it.

However, when `backend=None` (the default inline path), `LLMClient` manages its own retry loop internally — same as before. When a backend is injected, the expectation is that the backend handles its own retries for infrastructure-level concerns (e.g., Bedrock Batch job retries), and `LLMClient` handles application-level retries.

---

### Q: What data does InferenceResult carry?

```python
@dataclass
class InferenceResult:
    content: str                    # The LLM's response text
    finish_reason: str = "finished" # "finished" or "max_output_reached"
    usage: dict[str, int] = {}      # {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
    request_id: str = ""            # Correlates batch requests to results
```

The `usage` dict feeds directly into `CostHook` for token cost tracking. Backends that don't have token counts can leave it empty.

---

### Q: What happens if the external backend package isn't installed?

The factory raises `ImportError` at startup with a clear traceback showing which dotted path failed. This fails fast — you'll see the error in container logs immediately, not on the first inference call 10 minutes into processing.

---

### Q: Can I use different backends for different pipeline stages?

Not currently — the backend is set per `LLMClient` instance, and the factory creates one backend per pipeline. However, you could create multiple `LLMClient` instances with different backends in a custom orchestrator. The Protocol is stateless from the pipeline's perspective.

---

### Q: Where does the inference backend fit in the existing call chain?

```
ExtractionPipeline
  → ScoutRetrieval / ScoutChat
    → LLMClient.complete() / complete_batch()
      → [backend.infer() / backend.infer_batch()]    ← NEW: delegated here
        → litellm.acompletion()                       ← OR: inline path (when backend=None)
```

The backend sits between `LLMClient`'s message construction / retry logic and the actual LLM API call.

---

## For Architects

### Q: Why Protocol instead of ABC?

Three reasons:

1. **No coupling**: External backend projects don't need Scout AI as a dependency. They implement the method signatures and it works. An ABC would require `from scout_ai.inference.protocols import IInferenceBackend` — creating a dependency arrow from every backend project back to the seed.

2. **Structural subtyping**: Python Protocols use duck typing. If a class has `infer()` and `infer_batch()` with the right signatures, it satisfies the contract. No `class MyBackend(IInferenceBackend)` inheritance boilerplate.

3. **Codebase consistency**: `IPersistenceBackend` (persistence layer), `IPDFFormatter` (formatting layer) — all use the same Protocol pattern. Developers learn it once.

---

### Q: How does this compare to LangChain's LLM abstraction?

| Dimension | Scout AI | LangChain |
|-----------|----------|-----------|
| **Dependency** | Zero (Protocol, no import needed) | Heavy (langchain-core, pydantic v1 shims) |
| **Surface area** | 2 methods (`infer`, `infer_batch`) | 30+ methods, callbacks, streaming, structured output |
| **Batch support** | First-class (`infer_batch` on the Protocol) | Added later via separate `Runnable.batch()` |
| **Backend loading** | One env var, dotted-path | Registry, plugins, or manual wiring |
| **Vendor lock-in** | None (Protocol-based) | Tightly coupled to LangChain ecosystem |

Scout AI's approach is intentionally minimal. We don't need streaming, structured output, or callback chains for document extraction. Two methods cover 100% of the use case.

---

### Q: What about vendor lock-in?

There is none — by design. Scout AI doesn't import any backend except `RealTimeBackend`. External backends don't import Scout AI. The only coupling point is the method signature contract, which is 6 lines of Python.

If Ameritas decides to replace Scout AI entirely, the backend projects still work — they just need a new factory or direct instantiation. The inference backends are independent packages with their own lifecycle.

---

### Q: How does observability carry through different backends?

`CostHook` reads `InferenceResult.usage` regardless of which backend produced it. `AuditHook` logs at the `LLMClient` level, capturing model, latency, and token counts before and after the backend call. Backend-specific observability (Bedrock Batch job status, IDP service latency) is the backend's responsibility — it can emit its own logs, metrics, or traces using the `ObservabilityConfig` from `AppSettings`.

---

### Q: How does this affect the infrastructure deployment model?

The pluggable inference layer has a direct impact on deployment architecture:

**Same Docker image, different env vars**: Dev, staging, and production can use the exact same container image. The only difference is the `SCOUT_LLM_INFERENCE_BACKEND` env var in the task definition / deployment manifest. This means:
- One CI/CD pipeline builds one image
- ECS task definitions, EKS configmaps, and RHEL systemd units only differ in environment configuration
- No build-time branching for different inference strategies

**External backend packages are pip-installed alongside Scout AI**: The production Dockerfile installs `ameritas-bedrock-batch` as an additional pip dependency. The package is separate but deployed together:

```dockerfile
# Production Dockerfile
RUN pip install scout-ai ameritas-bedrock-batch
# ENV SCOUT_LLM_INFERENCE_BACKEND is set in the task definition, not the image
```

**Batch backends may need different compute profiles**: Real-time inference needs low-latency, always-on compute (Fargate, EKS pods). Batch inference may run as a scheduled job (ECS Scheduled Task, Kubernetes CronJob) with higher memory and no autoscaling. The pluggable layer means the application code is identical — only the deployment manifest changes.

---

### Q: How do we enable reusability across different projects?

The inference layer is reusable at three levels:

1. **Contract reusability**: Any Python project (not just Scout AI forks) can implement `IInferenceBackend`. A dental claims project, a disability claims project, and a compliance project all share the same inference contract. Backend packages built for one project work with all of them.

2. **Backend reusability**: The `ameritas-bedrock-batch` package isn't tied to APS or document extraction. It's a generic "submit inference jobs to Bedrock Batch and poll for results" implementation. The dental team, the retirement team, and the compliance team all use the same package.

3. **Seed reusability**: Teams forking Scout AI get the inference layer, persistence layer, domain system, and hook lifecycle for free. They write domain-specific code (questions, categories, synthesis) and choose their inference backend via env var.

```
                    ┌──────────────────────────┐
                    │  ameritas-bedrock-batch   │  ← shared backend package
                    └──────────┬───────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                    │
    ┌──────┴──────┐   ┌───────┴───────┐   ┌───────┴───────┐
    │ APS Project │   │ Dental Project│   │ Disability    │
    │ (Scout AI   │   │ (Scout AI     │   │ Project       │
    │  fork)      │   │  fork)        │   │ (Scout AI     │
    └─────────────┘   └───────────────┘   │  fork)        │
                                          └───────────────┘
```

---

## For Business Analysts / Product Owners

### Q: Does this change affect existing features?

No. The default behavior is `"realtime"` — exactly what runs today. All 416 existing tests pass without modification. No extraction results change, no PDF outputs change, no API responses change. The pluggable layer is additive.

---

### Q: What's the cost impact of Bedrock Batch?

AWS Bedrock Batch inference is typically **~50% cheaper** than real-time inference for the same model. For bulk document processing (ingesting 100+ APS documents), this translates directly to reduced per-document cost.

| Mode | Use Case | Relative Cost |
|------|----------|---------------|
| Real-time | Interactive extraction, single-document processing | 1x (baseline) |
| Bedrock Batch | Bulk processing, overnight ingestion, batch reports | ~0.5x |

The pluggable layer means adopting batch pricing is a configuration change, not a development project.

---

### Q: How does this affect project timelines?

For existing projects: **zero impact**. Nothing changes unless you opt in.

For new projects wanting batch support: instead of building custom Bedrock Batch integration (estimated 2-4 weeks), teams set one env var and install a shared backend package (estimated 1-2 hours). The backend package is built once and shared across all projects.

---

### Q: Which lines of business benefit?

Any LOB that uses LLM-powered document processing:

| LOB | Benefit |
|-----|---------|
| **Life Insurance (APS)** | Batch processing for bulk APS ingestion at reduced cost |
| **Dental Claims** | Same infrastructure, different domain questions |
| **Vision Claims** | Shared backend packages, consistent observability |
| **Disability** | IDP adapter for existing disability processing service |
| **Retirement Services** | Real-time for interactive, batch for bulk enrollment processing |
| **Compliance** | Batch processing for regulatory document review |

---

### Q: Can we A/B test different inference strategies?

Yes. Deploy two instances of the same service with different `SCOUT_LLM_INFERENCE_BACKEND` values and route traffic between them. The application code is identical — only the env var differs. This enables:
- Comparing real-time vs batch quality/latency
- Testing a new backend in production with a percentage of traffic
- Gradual rollout of batch processing

---

## For Leadership

### Q: What is the ROI of this architecture decision?

The ROI comes from three sources:

1. **Direct cost reduction**: Bedrock Batch is ~50% cheaper than real-time inference. For high-volume processing (thousands of documents/month), this is a significant line-item reduction.

2. **Development velocity**: Without the pluggable layer, every project that wants batch support builds its own integration (2-4 weeks each, repeated per project). With it, the first project builds the backend package (2-4 weeks once), and every subsequent project adopts it in hours.

3. **Reduced operational risk**: One contract means one testing pattern, one error handling strategy, one observability stack. Fewer surprises in production. Faster incident response because all AI services behave consistently.

---

### Q: How does this reduce risk?

| Risk | Mitigation |
|------|------------|
| **Vendor lock-in** | Backend is a pluggable module. Switching from Bedrock to SageMaker or a new provider is a new backend package, not a rewrite. |
| **Cascade failures** | Backend changes are isolated. A Bedrock Batch bug doesn't affect real-time processing. |
| **Knowledge silos** | One pattern across all projects. Engineers moving between teams don't relearn inference integration. |
| **Compliance** | Centralized audit logging via `AuditHook`. Every inference call is logged regardless of backend — consistent audit trail. |

---

### Q: What's the adoption path?

```
Phase 1 (Done)    — Pluggable inference layer shipped in Scout AI seed
Phase 2 (Next)    — Build ameritas-bedrock-batch package (infra team)
Phase 3           — APS production switches to batch for bulk ingestion
Phase 4           — Other LOBs fork Scout AI and inherit the capability
Phase 5           — IDP adapter built for teams with existing IDP services
```

Each phase is independent. Phase 2 doesn't block Phase 4. Teams can fork the seed today and get the pluggable layer even before the Bedrock Batch backend exists.

---

### Q: How does this fit the enterprise AI strategy?

Scout AI's pluggable layers (inference, persistence, domain, formatting) establish a **pattern language** for AI services at Ameritas:

- **Protocol contracts** define boundaries between teams
- **Dotted-path factories** enable runtime configuration without code changes
- **Env-var switching** makes deployment flexible across environments
- **Seed project model** means new AI initiatives start at 80% complete

This isn't just about document extraction. The inference layer pattern can be adopted by any AI service — chatbots, summarization engines, classification pipelines, recommendation systems. The contract is generic; the domain logic is specific.

---

### Q: What's the competitive advantage?

Speed to production. When a new business need requires AI-powered document processing, Ameritas doesn't start from scratch. The team:

1. Forks Scout AI (gets indexing, retrieval, extraction, inference, persistence, observability, deployment configs)
2. Writes domain-specific code (questions, categories, synthesis — the ~20% that's unique)
3. Sets env vars for their infrastructure (inference backend, persistence backend, LLM provider)
4. Deploys using existing Docker/ECS/EKS configs

**Weeks instead of months.** The pluggable inference layer is one piece of this, but it's a critical piece — it means the infrastructure team's investments (Bedrock Batch, IDP integration) automatically benefit every project that uses the seed.
