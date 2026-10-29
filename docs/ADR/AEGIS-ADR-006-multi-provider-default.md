# ADR-006: NVIDIA Default with OpenAI Alternative

- Status: Accepted
- Date: 2026-10-18
- Related: RFC-006, SRS-001 FR-009, C4-001

## Context

Aegis needs to support more than one LLM provider without coupling the API,
AgentService, or LangGraph execution path to a vendor SDK. NVIDIA is the
currently selected default provider; OpenAI remains available for explicit
selection and comparison.

Both providers support the generation and streaming operations required by
Aegis, but their credentials, model identifiers, endpoints, and SDK error
details are provider-specific. Generation usage metadata may also be absent.

## Decision

1. NVIDIA is the default provider when `LLM_PROVIDER` is not set.
2. OpenAI is supported as an alternative selected through `LLM_PROVIDER`.
3. `LLMProviderFactory` selects and constructs the configured adapter during
   application setup; the selected provider is injected into the existing
   AgentService and LangGraph path.
4. NVIDIA and OpenAI adapters implement the shared `LLMProvider` contract:
   `generate(request)` returns an `LLMResponse`, and `stream(request)` yields
   asynchronous text chunks.
5. Provider adapters map SDK responses and failures into provider-neutral
   models and Aegis exceptions. Usage fields are populated when supplied and
   remain optional when unavailable.
6. Provider-specific credentials, model IDs, and timeouts remain
   configuration-driven and must not be logged or committed.

## Rationale

- A shared interface keeps provider SDK types out of orchestration and API
  layers.
- Configuration-based selection permits provider comparison and replacement
  without changing agent execution code.
- The same contract enables deterministic adapter tests and fake-provider
  testing.
- Normalized errors let callers handle common failure classes consistently
  while adapters retain provider-specific diagnostics internally.

## Alternatives Considered

### OpenAI Only

Rejected because NVIDIA is the selected default and Aegis requires provider
portability and comparison.

### Provider-Specific Orchestration Paths

Rejected because separate paths would duplicate agent behavior and couple
LangGraph to vendor-specific APIs.

### Automatic Failover or Load Balancing

Deferred. The current decision supports explicit provider selection; it does
not define routing, retries across providers, or failover behavior.

## Consequences

- Each adapter must maintain compatible generation, streaming, error-mapping,
  and usage-mapping tests.
- Provider capabilities and available usage fields may differ; the common
  contract does not promise identical model behavior or metadata completeness.
- Live comparisons must record provider, model ID, sample count, and test
  conditions. A single run is evidence, not a general performance claim.
- The existing `/agent/stream` API currently simulates chunks after generation;
  it does not expose provider time-to-first-token behavior.

## References

- RFC-006: Multi-Provider Support
- SRS-001 FR-009: Multi-Provider LLM Requirements
- C4-001: Provider factory and adapter architecture
- EDI-001: Multi-provider traceability
- `src/aegis_platform/llm/factory.py`
- `src/aegis_platform/llm/providers/nvidia.py`
- `src/aegis_platform/llm/providers/openai.py`
- `tests/unit/llm/test_factory.py`
- `tests/unit/llm/test_nvidia_provider.py`
- `tests/unit/llm/test_openai_provider.py`
