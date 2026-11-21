# AEGIS-SRS-001

# Software Requirements Specification
>A SRS is a document that **describes what the software will do** and how it will be expected to perform. Acts as a contract between the client and the developer.

**Version:** 1.9
**Status:** Active
**Owner:** Kader Beevi
**Project:** Aegis AI Platform
**Last Updated:** 2026-11-20

## Related Artifacts

- PRD: AEGIS-PRD-001
- RFC-001: Platform Foundation
- RFC-002: FastAPI Foundation
- RFC-006: Multi-Provider Support
- RFC-007: LLM Provider Reliability
- RFC-008: LLM Provider Observability and Monitoring
- RFC-009: RAG Architecture
- ADR-001: Adopt FastAPI
- ADR-006: NVIDIA Default with OpenAI Alternative
- ADR-009: RAG Chunking Foundation
- C4: AEGIS-C4-001
- EDI: AEGIS-EDI-001

---

# Revision History

| Version | Date | Changes |
|----------|------|---------|
| 1.0 | 2026-09-27 | Initial platform requirements established. |
| 1.1 | 2026-09-28 | Added FastAPI API Foundation requirements, health endpoints, OpenAPI requirements, and implementation traceability. |
| 1.2 | 2026-09-29 | Added Agent Orchestration requirements, streaming responses, tool execution, and state management. |
| 1.3 | 2026-09-30 | Added LLM provider abstraction requirements for multi-provider support, synchronous and streaming invocation, model configuration, timeouts, and error normalization. |
| 1.5 | 2026-10-18 | Clarified configurable NVIDIA/OpenAI selection, shared generation and streaming contracts, normalized errors, usage metadata, and RFC-006 traceability. |
| 1.6 | 2026-11-02 | LLM Observability and Monitoring. |
| 1.7 | 2026-11-15 | RAG Foundation: added FR-013 (chunking, storage abstractions, factory pattern) and NFR-008 (provider independence, extensibility, configuration-driven behavior). |
| 1.8 | 2026-11-15 | RAG Phase 2 Implementation: added FR-014 (OpenAI EmbeddingProvider), FR-015 (Qdrant VectorStore), FR-016 (Factory extension), and NFR-009 (implementation quality: type safety, testing, error handling). |
| 1.9 | 2026-11-20 | Day 16: added FR-017 (RAG retrieval path contract, incl. FR-RAG-017-003 embedding usage) and FR-018 (retrieval safety layers, retroactive). Header version corrected (was 1.7). |

---

# Introduction

## 1.1 Purpose

This document defines the functional and non-functional software requirements for Aegis AI Platform.

It serves as the authoritative reference for implementation after product goals are defined in the PRD and technical proposals are evaluated through RFCs.

## 1.2 Scope

Aegis is a production-grade AI platform designed to demonstrate modern backend engineering practices.

The platform will eventually include:

- API Gateway
- Agent Orchestrator
- Retrieval-Augmented Generation (RAG)
- Event Streaming
- Evaluation Framework
- Observability
- Kubernetes Deployment

## 1.3 Intended Audience

- Software Engineers
- Architects
- Recruiters
- Interviewers
- Future Contributors

---

# 2. System Overview

## 2.1 High-Level Architecture

```text
Client
   │
   ▼
FastAPI Gateway
   │
   ├── Agent Service
   ├── Retrieval Service
   ├── Evaluation Service
   └── Future Services
```

## 2.2 Technology Baseline

| Area | Technology |
|------|------------|
| Language | Python 3.12 |
| API | FastAPI |
| Package Manager | uv |
| Validation | Pydantic |
| Testing | Pytest |
| Linting | Ruff |
| Type Checking | MyPy |
| Containers | Docker |
| CI | GitHub Actions |

---

# 3. Functional Requirements

## FR-001 — API Gateway

The platform shall expose a REST API through FastAPI.

Acceptance Criteria

- HTTP requests are accepted.
- Responses return valid JSON.
- OpenAPI documentation is generated automatically.

Related RFC

- RFC-002

---

## FR-002 — Health Monitoring

The platform shall expose production-ready health endpoints.

Required endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | Overall system health |
| `/health/live` | Liveness probe |
| `/health/ready` | Readiness probe |

Acceptance Criteria

- All endpoints return HTTP 200 during normal operation.
- Responses are machine-readable JSON.

---

## FR-003 — OpenAPI Documentation

The platform shall automatically generate API documentation.

Requirements

- Swagger UI available.
- ReDoc available.
- OpenAPI specification generated automatically.

---

## FR-004 — Request Validation

Incoming requests shall be validated using Pydantic.

Requirements

- Required fields enforced.
- Type validation performed.
- Invalid requests return appropriate HTTP responses.

---

## FR-005 — Future Agent Support

The platform shall support future AI agent integration without requiring API redesign.

Future integrations include:

- LangGraph
- OpenAI SDK
- Tool Calling
- Evaluation Engine

---

## FR-006 — Agent Orchestration

The platform shall execute multi-step workflows through LangGraph.

Requirements:

- StateGraph execution
- deterministic transitions
- retry capability

---

## FR-007 — Tool Execution

Agents shall invoke internal tools through standardized interfaces.

Requirements:

- typed inputs
- typed outputs
- isolated execution

---

## FR-008 — Streaming Responses

The platform shall support token streaming.

Acceptance Criteria:

- streaming endpoint
- incremental delivery
- graceful completion

---

## FR-009 — Multi-Provider LLM Requirements

The system shall:

- Select the LLM provider through configuration, defaulting to NVIDIA and supporting OpenAI as an alternative.
- Use `LLMProviderFactory` to create the selected adapter without changing the existing API → AgentService → LangGraph execution path.
- Require NVIDIA and OpenAI adapters to implement the common `LLMProvider` contract.
- Support generation through `generate(request) -> LLMResponse` and incremental asynchronous streaming through `stream(request) -> AsyncIterator[str]`.
- Use provider-neutral `LLMRequest`, `LLMResponse`, and `LLMUsage` models at the orchestration boundary.
- Return provider and model identifiers with generation results and include input, output, and total token usage when the provider supplies it; usage fields may be unavailable.
- Normalize provider SDK authentication, rate-limit, timeout, connection, and request failures into Aegis provider errors.
- Prevent API and agent orchestration layers from depending directly on provider SDKs.
- Prevent provider credentials from being stored in source control.
- Allow providers to be independently tested.
- Support future provider evaluation and comparison.

Acceptance Criteria

- NVIDIA is selected when `LLM_PROVIDER` is not supplied; OpenAI can be selected through configuration.
- Provider-specific model IDs, credentials, and timeout settings are read from configuration.
- Both adapters map generation content, provider, model, and available usage into `LLMResponse`.
- Both adapters yield non-empty streamed content incrementally and map SDK failures into provider-neutral errors.
- AgentService and LangGraph consume the shared interface and do not import provider SDKs.
- Unit tests cover provider selection, adapter generation, streaming, usage mapping, and normalized errors.

Related artifacts: RFC-006, ADR-006, C4-001, EDI-001.

---

## FR-010 - LLM Streaming and Reliability Requirements

 - The platform shall support incremental LLM output streaming.
 - The Agent Service shall consume LLM providers through a provider-independent abstraction.
 - The platform shall translate provider-specific failures into normalized platform errors.
 - The platform shall permit provider injection for automated tests.
 - The existing agent execution and calculator tool shall continue to function.

---

## FR-011 - Provider Reliability
- **FR-LLM-REL-001:** Configure maximum total attempts.
- **FR-LLM-REL-002:** Retry only explicitly classified transient failures.
- **FR-LLM-REL-003:** Do not retry authentication or invalid-request failures.
- **FR-LLM-REL-004:** Enforce an overall deadline across attempts and backoff.
- **FR-LLM-REL-005:** Propagate cancellation without retry.
- **FR-LLM-REL-006:** Retry a stream only before the first chunk is delivered.
- **FR-LLM-REL-007:** Never transparently restart a stream after partial output.
- **FR-LLM-REL-008:** Record retry metadata without secrets or full prompt content.

---

## FR-012 - LLM Observability
- **OBS-FR-001** — Request Count
   - The platform shall record one logical request count for each generation or streaming operation.

- **OBS-FR-002** — Failure Count
   - The platform shall record failed logical LLM operations.

- **OBS-FR-003** — Request Duration
   - The platform shall measure end-to-end LLM request duration, including retry delays.

- **OBS-FR-004** — Streaming TTFT
   - The platform shall measure time to first non-empty streaming chunk.

- **OBS-FR-005** — Stream Interruption
   - The platform shall record interrupted streams, including failures after partial output.

- **OBS-FR-006** — Token Usage
   - The platform shall record input and output token usage when returned by the provider.

- **OBS-FR-007** — Retry Telemetry
   - The retry mechanism shall expose a callback for each scheduled retry.

- **OBS-FR-008** — Structured Events
   - The platform shall emit structured events for successful requests, failures, retries, and stream interruptions.

---

## FR-013 - RAG Foundation (Chunking & Storage Abstractions)

The platform shall support Retrieval-Augmented Generation with provider-agnostic abstractions for text processing and vector storage.

### FR-RAG-001 — Text Chunking
- The platform shall support multiple text chunking strategies (recursive character-based, markdown-aware).
- Chunking shall be configurable via `ChunkingConfig` with strategy name, chunk size, overlap, and size limits.
- Each chunk shall track position in original document, sequence number, and metadata (strategy, document_id).
- The platform shall support at least 2 LangChain-based chunking strategies: recursive and markdown.

### FR-RAG-002 — Chunking Configuration
- Chunking strategy shall be selectable at runtime via configuration without code changes.
- Configuration parameters: strategy name, chunk_size (target), overlap (between chunks), min_chunk_size, max_chunk_size.
- Configuration shall be immutable after chunker instantiation; updates require re-instantiation.

### FR-RAG-003 — Embedding Provider Abstraction
- The platform shall define an `EmbeddingProvider` protocol supporting provider-neutral embedding generation.
- Embedding providers shall not be locked to specific vendor SDKs in the core RAG module.
- Support for future implementations: OpenAI, Claude, local embeddings without modifying existing code.

### FR-RAG-004 — Vector Storage Abstraction
- The platform shall define a `VectorStore` protocol supporting provider-neutral vector search.
- Vector storage shall support query, insert, update, and delete operations.
- Vector stores shall not be locked to specific vendors in the core RAG module.
- Support for future implementations: Qdrant, Weaviate, Pinecone without modifying existing code.

### FR-RAG-005 — Factory Pattern & Dependency Injection
- The platform shall provide a factory function `create_chunker(config)` for instantiating chunking strategies.
- Strategy registry shall map configuration names to strategy classes.
- Available strategies shall be discoverable via `get_available_strategies()`.

### FR-RAG-006 — Document Models
- The platform shall define data models for: Document, Chunk, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, SearchResult (embedding request/response shape per ADR-013).
- Chunk model shall include: text, chunk_id, sequence, start_pos, end_pos, metadata.
- Document model shall include: doc_id, content, metadata, source.

Acceptance Criteria:

- LangChain-based recursive and markdown chunkers are implemented and tested.
- Abstract protocols (EmbeddingProvider, VectorStore) are defined but not implemented.
- Factory function correctly instantiates chunkers based on configuration.
- All 24 unit tests pass covering chunking, configuration, factory, and integration scenarios.
- Type checking passes (mypy).
- Linting passes (ruff).
- Module documentation (README.md) provides architecture overview and usage examples.

---

## FR-014 - OpenAI EmbeddingProvider Implementation (Phase 2)

The platform shall implement a production-ready OpenAI embedding provider.

### FR-RAG-014-001 — OpenAI Embeddings
- The platform shall support OpenAI text-embedding-3-small model (1536-dimensional vectors).
- Configuration shall include API key (from environment), model, dimension, timeout, retries, and batch size.
- The provider shall support async embedding of single texts and batches (up to 100 texts per request).
- Auto-batching shall split larger inputs into multiple requests transparently.
- Rate limit handling (HTTP 429) shall implement exponential backoff with jitter.

### FR-RAG-014-002 — Error Handling
- Authentication errors (401) shall fail immediately with clear error messages.
- Connection errors shall retry according to max_retries configuration.
- Timeout errors shall fail gracefully.
- Empty or invalid input shall be handled without crashing.

### FR-RAG-014-003 — Health Checks
- The provider shall support health checks to verify OpenAI API connectivity.
- Health check failures shall not block normal operations.

Acceptance Criteria:

- `OpenAIEmbedding` class fully implements `EmbeddingProvider` protocol.
- 12 unit tests cover: configuration, single embedding, batch embedding, rate limiting, auth errors, timeouts, edge cases.
- 2 integration tests cover: end-to-end chunk→embed→store, concurrent embeddings.
- Type checking passes (mypy --strict).
- Linting passes (ruff).
- All tests mocked (no real OpenAI API calls in CI/CD).

---

## FR-015 - Qdrant VectorStore Implementation (Phase 2)

The platform shall implement a production-ready Qdrant vector database backend.

### FR-RAG-015-001 — Vector Store Operations
- The platform shall support collection management (create, delete, list).
- The platform shall support CRUD operations: insert vectors with metadata, search by similarity, delete by ID, update vectors and metadata.
- Search shall support limiting results and score thresholds.
- Collection statistics shall be available (vector count, dimension, distance metric).

### FR-RAG-015-002 — Configuration
- Configuration shall include URL (default: http://localhost:6333), optional API key, timeout, embedding dimension, and distance metric.
- Distance metrics shall include cosine (default), euclidean, and dot product.

### FR-RAG-015-003 — Error Handling
- Missing collections shall fail with descriptive error messages.
- Invalid vector dimensions shall be validated before operations.
- Connection errors shall fail gracefully with clear messages.

### FR-RAG-015-004 — Health Checks
- The store shall support health checks to verify Qdrant connectivity.
- Health check failures shall not block normal operations.

Acceptance Criteria:

- `QdrantVectorStore` class fully implements `VectorStore` protocol.
- 12 unit tests cover: configuration, collection management, insert, search, delete, update, error handling, edge cases.
- 2 integration tests cover: end-to-end create→insert→search, concurrent operations.
- Type checking passes (mypy --strict).
- Linting passes (ruff).
- All tests mocked (no real Qdrant calls in CI/CD).

---

## FR-016 - Factory Extension for Embedding & Vector Store Providers (Phase 2)

The platform shall extend the factory pattern to support embedding providers and vector stores.

### FR-RAG-016-001 — Provider Registries
- The platform shall maintain registries mapping provider names to implementation classes.
- Registries shall be extensible for future providers without modifying existing code.

### FR-RAG-016-002 — Factory Functions
- `create_embedding_provider(config)` shall instantiate embedding providers based on configuration.
- `create_vector_store(config)` shall instantiate vector stores based on configuration.
- Both functions shall validate configuration and raise clear errors for unknown providers.

### FR-RAG-016-003 — Provider Discovery
- `get_available_embedding_providers()` shall return list of available provider names.
- `get_available_vector_stores()` shall return list of available store names.

### FR-RAG-016-004 — Configuration-Driven Selection
- Providers shall be selectable via environment variables without code changes.
- Default provider selections shall support sensible defaults.

Acceptance Criteria:

- Factory functions fully implemented and tested.
- 6 unit tests cover: provider creation, validation, error handling, discovery.
- 2 integration tests cover: environment variable configuration, end-to-end usage.
- Type checking passes (mypy --strict).
- Linting passes (ruff).

---

## FR-017 - RAG Retrieval Path Contract (Day 16, ADR-012)

### FR-RAG-017-001 — Embedder Conformance
- Every embedding provider shall implement `embed_text(str) -> np.ndarray` and
  `embed_batch(list[str]) -> list[np.ndarray]`, with output order and length matching the input.
- Batches larger than the provider limit shall be split into several API calls.
- Asymmetric models shall receive `input_type` = `query` for searches and `passage` for ingested text.
- Only transient failures (rate limit, 5xx, connection) shall be retried; client errors and quota
  exhaustion shall fail immediately without exposing response bodies.

### FR-RAG-017-003 — Embedding Usage (ADR-013)
- `EmbeddingProvider.embed(EmbeddingRequest) -> EmbeddingResponse` shall return vectors, the serving model
  and token usage summed over API calls; usage shall be `None` when any call does not report it.
- Ingestion and search results shall include `embedding_tokens`.

### FR-RAG-017-002 — Vector Store Behavior
- Point IDs shall be deterministic across processes; re-ingesting a chunk shall overwrite it.
- Re-ingesting a document shall replace all of its chunks; chunks from an older, longer version shall be
  removed, and a failed ingest shall leave the previous version searchable.
- The collection shall be created on first insert with the embedding dimension and configured metric;
  existing collections shall never be dropped implicitly.
- Searching before any ingestion shall return no results rather than an error.
- The collection shall be configurable (`RAG_COLLECTION`). A collection shall record the embedding model,
  dimension and metric it was built with, and operations shall be refused when the current configuration
  differs, so vectors from different models are never mixed.
- `/rag/health` shall fail when the embedder or the vector store is unavailable.
- `/rag/collections` shall return name, vector size, point count and distance for each collection.

Acceptance Criteria:

- Real qdrant-client (local mode) tests cover ingest, search, re-ingest, >100 chunks, collections, health.
- Mocked-HTTP tests cover NVIDIA `input_type`, batching, fail-fast and retry behavior.

---

## FR-018 - RAG Retrieval Safety Layers (Day 12, recorded Day 16, ADR-012)

- Queries shall pass length and injection guardrails before retrieval; LLM query expansions shall be
  re-checked.
- Results shall be filtered by per-chunk access level, allowed roles and organization.
- Organization-scoped, non-public documents shall be readable only by callers of the same organization,
  including callers that send no organization.
- Returned text shall be PII-masked; retrieval shall fail closed if masking is unavailable.
- Optional cross-encoder reranking shall be available via `?rerank=true`.
- `/rag/verify` shall report citation coverage and hallucination risk for a supplied answer.

---

# 4. Non-Functional Requirements

## NFR-001 — Performance

The platform should support asynchronous request handling.

Target characteristics

- Low-latency responses
- Efficient concurrent request handling
- Non-blocking I/O

---

## NFR-002 — Reliability

The platform should support resilient operation.

Future capabilities

- Retry mechanisms
- Circuit breakers
- Timeouts
- Graceful degradation

---

## NFR-003 — Security

Security requirements include:

- JWT authentication (future)
- Secure secret management
- Input validation
- Safe error handling

Sensitive information must never appear in logs.

---

## NFR-004 — Observability

Every service should support observability.

Minimum expectations

- Structured logging
- Metrics
- Distributed tracing
- Health endpoints

Future implementation

- OpenTelemetry
- Prometheus
- Grafana

---

## NFR-005 — Maintainability

The platform shall prioritize long-term maintainability.

Requirements

- Type hints
- Modular architecture
- Consistent naming
- Documentation-first workflow

---

## NFR-006 - Provider Reliability
- **NFR-LLM-REL-001:** Bound latency with a configurable deadline.
- **NFR-LLM-REL-002:** Keep policy provider-neutral.
- **NFR-LLM-REL-003:** Make behavior testable using fake providers and injected timing.
- **NFR-LLM-REL-004:** Keep retry logic at one composition boundary.
- **NFR-LLM-REL-005:** Keep credentials and full prompts out of logs.

---

## NFR-007 - LLM Observability

- **OBS-NFR-001** — Provider Independence
   - The observability implementation shall not depend on OpenAI or NVIDIA SDK types.

- **OBS-NFR-002** — Privacy
   - Prompts, completions, API keys, user IDs, and raw exception messages shall not be emitted.

- **OBS-NFR-003** — Bounded Cardinality
   - Metric labels shall be restricted to bounded dimensions such as provider and model.

- **OBS-NFR-004** — Failure Isolation
   - Telemetry sink failures shall not affect LLM request execution.

- **OBS-NFR-005** — Cancellation
   - Async cancellation shall propagate correctly.

- **OBS-NFR-006** — Testability
   - Metrics and event sinks shall be replaceable with in-memory implementations for unit testing.

---

## NFR-008 - RAG Provider Independence

- **RAG-NFR-001** — Embedding Provider Neutrality
   - The RAG module shall not depend on specific embedding provider SDKs in the core protocol definitions.
   - Embedding providers shall be selectable at runtime via configuration.

- **RAG-NFR-002** — Vector Store Neutrality
   - The RAG module shall not depend on specific vector store SDKs in the core protocol definitions.
   - Vector stores shall be selectable at runtime via configuration.

- **RAG-NFR-003** — Extensibility
   - The protocol-based architecture shall allow new chunking strategies, embedding providers, and vector stores to be added without modifying the core module.

- **RAG-NFR-004** — Configuration-Driven Behavior
   - Chunking strategy, chunk size, overlap, and size limits shall be configurable without code changes.
   - Configuration parameters shall be validated at instantiation time.

- **RAG-NFR-005** — Immutable Configuration After Instantiation
   - Chunker configuration shall be immutable after instantiation to prevent inconsistent state.
   - Configuration updates shall require re-instantiation of the chunker.

---

## NFR-009 - RAG Embedding & Storage Implementation Quality (Phase 2)

- **RAG-IMPL-NFR-001** — Type Safety
   - All new code shall pass mypy --strict type checking.
   - All parameters and return types shall be fully annotated.

- **RAG-IMPL-NFR-002** — Code Quality
   - All new code shall pass ruff linting with no violations.
   - Code formatting shall be consistent with ruff format.
   - Docstrings shall be present and complete for all public APIs.

- **RAG-IMPL-NFR-003** — Test Coverage
   - Implementation shall include 30+ unit and integration tests.
   - Test coverage shall exceed 85% for all new code.
   - All external API calls (OpenAI, Qdrant) shall be mocked in tests.

- **RAG-IMPL-NFR-004** — Error Handling & Retry
   - Rate limiting (HTTP 429) shall implement exponential backoff with jitter.
   - Connection errors shall retry according to configuration.
   - Authentication errors shall fail immediately without retry.
   - All errors shall have descriptive messages for operators.

- **RAG-IMPL-NFR-005** — Configuration & Secrets
   - All secrets (API keys) shall come from environment variables.
   - All configuration shall support sensible defaults.
   - Configuration shall be validated at instantiation time.
   - No hardcoded credentials shall appear in source code.

- **RAG-IMPL-NFR-006** — Performance
   - Embedding batch operations shall complete in <1 second for 100 vectors.
   - Vector search in <1M vectors shall complete in <50ms (indexed).
   - Metadata filtering shall not significantly impact search performance.

---

# 5. Interface Requirements

## REST API

The API shall:

- accept JSON
- return JSON
- expose OpenAPI
- use HTTP status codes consistently

## Internal Services

Future services shall communicate through well-defined interfaces.

Examples

- Kafka events
- Internal REST APIs
- Background workers

---

# 6. Data Requirements

Current Phase

No persistent application data is required.

Future phases introduce:

- PostgreSQL
- Qdrant
- Kafka event storage

Data architecture will be expanded through future RFCs and ADRs.

---

# 7. Operational Requirements

## Local Development

Developers shall be able to run the platform locally using:

- uv
- FastAPI
- Docker

## Continuous Integration

Every push should execute:

- Ruff
- MyPy
- Pytest

Future deployment targets

- AKS
- Amazon EKS

---

# 8. Traceability Matrix

| Requirement | RFC | ADR |
|-------------|-----|-----|
| API Gateway | RFC-002 | ADR-001 |
| Health Endpoints | RFC-002 | ADR-001 |
| OpenAPI | RFC-002 | ADR-001 |
| Request Validation | RFC-002 | ADR-001 |
| Python Foundation | RFC-002 | ADR-001 |
| Multi-provider selection, common generation/streaming, normalized errors, and usage metadata | RFC-006 | ADR-006 |
| LLM streaming reliability | RFC-005 | ADR-005 |
| RAG Chunking Foundation | FR-RAG-001 to FR-RAG-006, NFR-RAG-001 to NFR-RAG-005 | RFC-009 | ADR-009 |

---

# 9. Acceptance Criteria

The current implementation phase is complete when:

- [ ] Python environment is configured.
- [ ] FastAPI initializes successfully.
- [ ] `/health` returns HTTP 200.
- [ ] `/health/live` works.
- [ ] `/health/ready` works.
- [ ] Swagger UI loads.
- [ ] Ruff passes.
- [ ] MyPy passes.
- [ ] Pytest passes.
- [ ] Docker builds successfully.
- [ ] GitHub Actions passes.

---

# 10. Future Requirements

Future SRS revisions will add requirements for:

- LangGraph orchestration
- RAG pipeline
- Kafka event streaming
- PostgreSQL persistence
- Qdrant vector search
- Evaluation framework
- Multi-agent workflows
- Production
- multi-agent
- memory management
- human approval workflows

---

# 11. Verification
Test success, recovery, exhaustion, permanent failures, deadline, cancellation, and stream failure before/after first chunk. Run NVIDIA/OpenAI contract tests without requiring live credentials.

---

# 12. Validation

Automated tests shall verify:

Successful generation.

Failed generation.

Successful streaming.

Streaming TTFT.

Stream interruption after partial output.

Token usage recording.

Privacy-safe event attributes.

Observability sink failure isolation.

---

# Engineering Note

This SRS evolves incrementally.

Every accepted RFC that changes software behavior must result in an SRS revision, preserving a complete engineering history through the Revision History and Traceability Matrix.
