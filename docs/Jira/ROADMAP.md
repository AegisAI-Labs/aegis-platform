# AEGIS Roadmap

Version: 2.0
Last Updated: 2026-11-20 (Day 16 — RAG real path)

The original 16-week roadmap is the source of truth. Work added outside it is tracked
under **Extensions** and does not replace any milestone.

Status values are based on source code and test evidence, not on design documents:
**Done**, **Partial** (implemented with known gaps), **Not started**.

---

# 16-Week Roadmap — Status

| Week | Milestone | Status | Evidence / known gaps |
|------|-----------|--------|-----------------------|
| 1 | FastAPI Foundation | Done | `api/main.py`, health/ready/live, CI, pre-commit, tag `v0.1.0`. Exception handlers added Day 15. Gap: no app factory (Day 18). |
| 2 | OpenAI / LLM Integration | Done | `llm/` protocol, models, errors, factory, OpenAI provider; unit tests. |
| 3 | LangGraph Agent Architecture | Done | `agent/graph.py` route → calculator / llm. Day 15 (ADR-011): real streaming through the graph, AST-safe calculator, whole-word routing with expression extraction, structured HTTP/SSE error mapping. |
| 4 | RAG | Partial | Chunking, OpenAI/NVIDIA embeddings, Qdrant store, `/rag/*` endpoints, safety layers. Day 16 (ADR-012): embedder contract, Qdrant `query_points`, deterministic IDs, auto-created collection, honest health, working `/rag/collections`, tenant fix. Remaining: grounded answer generation and agent integration (Day 17). |
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
| Retry reliability (`RetryingLLMProvider`) | 7, 14 | Done | ADR-007 (Accepted). Day 14: stream retries emit retry telemetry; SDK retries disabled (`max_retries=0`); policy in `Settings` (`LLM_RETRY_*`). |
| LLM observability (`ObservedLLMProvider`) | 8, 14 | Done | RFC-008 / ADR-008 (Accepted). Day 14: model label fallback, cancellation/early-close semantics, interruption only after output, `X-Correlation-ID` middleware, bounded sinks, all 15 required tests. Export to Prometheus/OTel is W9/W10. |
| RAG safety layers | 12, 16 | Partial | ADR-012, SRS FR-018 (recorded Day 16). Tenant gap fixed Day 16. Remaining: blocking reranker/Presidio calls in async code (Day 17). |

---

# Stabilization Plan (before any new milestone)

No new milestone (W5 PostgreSQL, W6 Kafka) starts until these days are committed and all checks pass.

| Day | Scope | Status |
|-----|-------|--------|
| 13 | Baseline hygiene + roadmap status reconciliation | Done |
| 14 | Close LLM observability milestone | Done |
| 15 | Agent hardening (real streaming, error mapping, safe calculator, routing) | Done |
| 16 | RAG real path (embedder interface, Qdrant query/IDs/collection, tests) | Done |
| 17 | RAG grounded answers + agent integration (RFC-009 Phase 3) | Planned |
| 18 | Security and runtime hardening | Planned |
| 19 | Documentation reconciliation, W4 close-out, `v0.2.0` | Planned |
