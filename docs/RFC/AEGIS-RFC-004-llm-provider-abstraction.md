# RFC-004: LLM Provider Abstraction

- Status: Proposed
- Date: 2026-09-30
- Owner: Aegis Engineering
- Related: PRD, SRS, C4 Architecture, ADR-004, EDI

---

# 1. Summary

Aegis requires an abstraction layer that allows the agent platform
to interact with multiple Large Language Model providers without
coupling agent orchestration, API contracts, or business logic to
a specific provider SDK.

The provider abstraction will support synchronous generation,
streaming generation, model configuration, error normalization,
timeouts, and provider-specific metadata.

---

# 2. Problem

Directly integrating an LLM SDK into the Agent Service or LangGraph
creates tight coupling between Aegis and a specific provider.

For example:

    Agent → OpenAI SDK

would make provider replacement or multi-provider support expensive.

Aegis should instead implement:

    Agent
       ↓
    LLM Provider Interface
       ↓
    Provider Adapter
       ↓
    Provider SDK

---

# 3. Goals

- Support multiple LLM providers.
- Keep agent orchestration provider-independent.
- Keep FastAPI contracts provider-independent.
- Support synchronous generation.
- Support streaming generation.
- Normalize provider errors.
- Support configurable models.
- Support configurable timeouts.
- Support provider-specific metadata without leaking provider
  implementation details.
- Make providers independently testable.
- Enable future provider comparison and evaluation.

---

# 4. Non-Goals

The following are not required in the initial implementation:

- Fine-tuning models.
- Training models.
- Automatic provider selection based on quality.
- Complex multi-agent routing.
- Production-scale cost optimization.
- Full semantic caching.
- Advanced model routing.

These may be introduced in later phases.

---

# 5. Proposed Architecture

    FastAPI
       ↓
    Agent Service
       ↓
    LangGraph
       ↓
    LLM Provider Interface
       ↓
    Provider Factory
       ↓
    Provider Adapter
       ↓
    Provider SDK/API

Example:

    ┌──────────────────────┐
    │     Agent Service    │
    └──────────┬───────────┘
               ↓
    ┌──────────────────────┐
    │ LLM Provider Interface│
    └──────────┬───────────┘
               ↓
         Provider Factory
               ↓
      ┌────────┼────────┐
      ↓        ↓        ↓
   Provider A Provider B Provider C

---

# 6. Provider Contract

The initial abstraction should support:

- generate()
- stream()

Future capabilities may include:

- structured output
- tool calling
- embeddings
- multimodal input
- token usage
- model metadata

---

# 7. Configuration

Provider configuration must come from application configuration
rather than hard-coded values.

Examples:

    LLM_PROVIDER
    LLM_MODEL
    LLM_TIMEOUT_SECONDS
    LLM_MAX_RETRIES

Secrets must never be committed to source control.

---

# 8. Error Handling

Provider-specific exceptions should be normalized into Aegis
domain-level errors.

Examples:

- ProviderUnavailable
- ProviderTimeout
- ProviderRateLimited
- ProviderAuthenticationError
- ProviderRequestError

The API layer should not depend on provider SDK exceptions.

---

# 9. Streaming

The abstraction must support streaming so that:

    POST /agent/stream

does not need to know whether the underlying provider is OpenAI,
Anthropic, Gemini, a local model, or another implementation.

---

# 10. Testing Strategy

Testing must occur at multiple levels:

- Provider unit tests
- Provider contract tests
- Agent integration tests
- API tests
- Streaming tests
- Failure tests
- Configuration tests

Real provider calls should not be required for the normal unit-test
suite.

---

# 11. Experiments

The implementation should allow comparison of:

- Provider latency
- Streaming behavior
- Error handling
- Response consistency
- Token usage where available
- Provider-specific limitations

Results should be documented as engineering evidence.

---

# 12. Security

Provider credentials must be supplied through environment variables
or a secure secret-management mechanism.

Secrets must not appear in:

- Git
- logs
- test fixtures
- exception messages
- telemetry attributes

---

# 13. Future Extensions

Potential future capabilities:

- Provider fallback
- Model routing
- Cost-aware routing
- Semantic caching
- Rate limiting
- Provider health checks
- Circuit breakers
- Multi-region provider routing
- Model evaluation
