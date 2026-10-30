# AEGIS-SRS-001

# Software Requirements Specification
>A SRS is a document that **describes what the software will do** and how it will be expected to perform. Acts as a contract between the client and the developer.

**Version:** 1.5
**Status:** Active
**Owner:** Kader Beevi
**Project:** Aegis AI Platform
**Last Updated:** 2026-10-18

## Related Artifacts

- PRD: AEGIS-PRD-001
- RFC-001: Platform Foundation
- RFC-002: FastAPI Foundation
- RFC-006: Multi-Provider Support
- ADR-001: Adopt FastAPI
- ADR-006: NVIDIA Default with OpenAI Alternative
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

## NFR - Provider Reliability
- **NFR-LLM-REL-001:** Bound latency with a configurable deadline.
- **NFR-LLM-REL-002:** Keep policy provider-neutral.
- **NFR-LLM-REL-003:** Make behavior testable using fake providers and injected timing.
- **NFR-LLM-REL-004:** Keep retry logic at one composition boundary.
- **NFR-LLM-REL-005:** Keep credentials and full prompts out of logs.

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

# Engineering Note

This SRS evolves incrementally.

Every accepted RFC that changes software behavior must result in an SRS revision, preserving a complete engineering history through the Revision History and Traceability Matrix.
