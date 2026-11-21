# AEGIS Roadmap

Version: 2.0
Last Updated: 2026-11-20 (Day 13 — status reconciliation)

The original 16-week roadmap is the source of truth. Work added outside it is tracked
under **Extensions** and does not replace any milestone.

Status values are based on source code and test evidence, not on design documents:
**Done**, **Partial** (implemented with known gaps), **Not started**.

---

# 16-Week Roadmap — Status

| Week | Milestone | Status | Evidence / known gaps |
|------|-----------|--------|-----------------------|
| 1 | FastAPI Foundation | Done | `api/main.py`, health/ready/live, CI, pre-commit, tag `v0.1.0`. Gaps: no app factory, no exception handlers. |
| 2 | OpenAI / LLM Integration | Done | `llm/` protocol, models, errors, factory, OpenAI provider; unit tests. |
| 3 | LangGraph Agent Architecture | Done (defects) | `agent/graph.py` route → calculator / llm. Gaps: calculator uses `eval`; keyword routing matches "address" as "add"; `/agent/stream` replays a finished answer instead of streaming. |
| 4 | RAG | Partial | Chunking, OpenAI/NVIDIA embeddings, Qdrant store, `/rag/*` endpoints, safety layers. Gaps: OpenAI embedder does not match the `EmbeddingProvider` interface; Qdrant `search` call not available in qdrant-client 1.20; unstable point IDs; collection never created; no grounded answer generation; not integrated into the agent. |
| 5 | PostgreSQL | Not started | Persistence for conversations, execution history and the RAG document registry. |
| 6 | Kafka | Not started | Event-driven processing (producers/consumers) behind the `EventSink` contract. |
| 7 | Docker | Partial | App `Dockerfile` (root user, no HEALTHCHECK); `docker-compose.qdrant.yml` only. |
| 8 | Kubernetes | Not started | |
| 9 | OpenTelemetry | Not started | In-memory metrics/events only (see Extensions). |
| 10 | Grafana | Not started | |
| 11 | AI Evaluation | Not started | `evaluations/` is a placeholder. |
| 12 | Load Testing | Not started | |
| 13 | AWS | Not started | |
| 14 | CI/CD | Partial | GitHub Actions: locked sync, ruff check, ruff format check, mypy, pytest, pre-commit. No image build or deploy. |
| 15 | Open Source Readiness | Partial | CONTRIBUTING, CODEOWNERS, SECURITY, LICENSE exist; several are stale. |
| 16 | Portfolio Polish | Not started | |

---

# Extensions (added work, tracked separately)

| Extension | Day | Status | Evidence / known gaps |
|-----------|-----|--------|-----------------------|
| LLM streaming + error boundaries | 5 | Done | RFC-005, ADR-005. |
| Multi-provider (OpenAI + NVIDIA) | 6 | Done | RFC-006, ADR-006, `llm/providers/nvidia.py`. |
| Retry reliability (`RetryingLLMProvider`) | 7 | Done (defects) | ADR-007. Gaps: stream retries emit no retry callback; SDK default retries stack with ours. |
| LLM observability (`ObservedLLMProvider`) | 8 | Partial | RFC-008, ADR-008. Gaps: model label `"None"`; cancellation telemetry; correlation-ID middleware; unbounded sinks; 4 of 15 required tests. |
| RAG safety layers | 12 | Partial | Guardrails, PII, access control, rerank, citations, hallucination checks. Gaps: no RFC/ADR/SRS entry; blocking calls in async code. |

---

# Stabilization Plan (before any new milestone)

No new milestone (W5 PostgreSQL, W6 Kafka) starts until these days are committed and all checks pass.

| Day | Scope | Status |
|-----|-------|--------|
| 13 | Baseline hygiene + roadmap status reconciliation | Done |
| 14 | Close LLM observability milestone | Planned |
| 15 | Agent hardening (real streaming, error mapping, safe calculator, routing) | Planned |
| 16 | RAG real path (embedder interface, Qdrant query/IDs/collection, tests) | Planned |
| 17 | RAG grounded answers + agent integration (RFC-009 Phase 3) | Planned |
| 18 | Security and runtime hardening | Planned |
| 19 | Documentation reconciliation, W4 close-out, `v0.2.0` | Planned |
