# AEGIS API Reference — RAG Endpoints and Safety Modules

**Scope:** RFC-009 Phase 2a/2b (TASK-048)
**Related:** [Configuration guide](../GUIDES/RAG-SAFETY-CONFIGURATION.md) · [Usage guide](../GUIDES/RAG-SAFETY-USAGE.md) · [RFC-009](../RFC/AEGIS-RFC-009-RAG-Architecture.md) · [ADR-010](../ADR/AEGIS-ADR-010-RAG-Service-Wiring.md)

This document has two parts:

1. [REST endpoints](#1-rest-endpoints) exposed by the FastAPI app.
2. [Python API](#2-python-api--safety-modules) of the safety modules in `aegis_platform.rag`.

> **Safety layers.** `/rag/search` and `/rag/query` run guardrails, access control and PII
> masking on every call; query enhancement (`?enhance=true`) and cross-encoder reranking
> (`?rerank=true`) are opt-in. `/rag/verify` runs citation tracking and hallucination
> detection on an answer. The pipeline lives in `aegis_platform.rag.secure_retrieval`
> ([section 2.8](#28-safety-pipeline--secure_retrieval)).

---

## 1. REST endpoints

The RAG router is mounted at **`/rag`** (`src/aegis_platform/api/main.py`). Interactive OpenAPI
docs are served at `http://localhost:8000/docs` while the app is running.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/rag/health` | Check embedding provider and vector store connectivity |
| `POST` | `/rag/embed` | Embed a single text |
| `POST` | `/rag/ingest` | Chunk, embed and store documents with access metadata |
| `POST` | `/rag/search` | Semantic search through the safety pipeline |
| `POST` | `/rag/query` | Retrieval step of RAG through the safety pipeline |
| `GET` | `/rag/collections` | List vector store collections |
| `POST` | `/rag/verify` | Citation and hallucination check of an LLM answer |

All documents are stored in, and searched from, a single collection named `documents`.

### Caller identity

Access control needs to know who is calling. The API reads three **trusted** headers:

| Header | Example | Notes |
|---|---|---|
| `X-User-Id` | `alice` | Required together with `X-User-Role` |
| `X-User-Role` | `analyst` | `admin`, `analyst`, `researcher`, `user`, `guest` (case-insensitive); anything else returns `400` |
| `X-Organization-Id` | `acme` | Tenant; strongly recommended |

Without `X-User-Id` and `X-User-Role`, the caller is an anonymous **guest**: `/search` and
`/query` return only `PUBLIC` documents, and `/ingest` returns `403`.

> The API does not authenticate these headers. Deploy it behind a gateway that authenticates
> the user, strips any client-supplied `X-User-*` / `X-Organization-Id` headers, and sets them
> itself. Exposed directly, any client can claim to be an admin.

### Status codes

| Code | When |
|---|---|
| `200` | Success |
| `400` | Invalid input (`ValueError`), unknown `X-User-Role`, or query rejected by guardrails |
| `403` | Guest ingestion, or a non-admin ingesting into another organization |
| `422` | Request body failed schema validation (blank string, `top_k` out of range, too long) |
| `503` | Provider or vector store failure, PII masking unavailable, or the service could not be initialised |

Error bodies use FastAPI's format: `{"detail": "<message>"}`. For `422`, `detail` is a list of
validation errors. A guardrail rejection returns a structured `detail`:

```json
{
  "detail": {
    "message": "Query rejected by guardrails",
    "risk_level": "HIGH",
    "violations": [
      "Detected command injection pattern: shell command after separator",
      "Detected command injection pattern: destructive rm flags"
    ]
  }
}
```

The service is created on the first request and cached (`@lru_cache`). If configuration is
missing (for example no embedding key), **every endpoint that needs the service returns `503`**
with `RAG Service initialization failed: ...`. `/rag/verify` does not need it.

### `POST /rag/health`

No request body.

```json
{
  "status": "healthy",
  "embedding_provider": "OpenAIEmbedding",
  "embedding_model": "unknown",
  "vector_store": "QdrantVectorStore",
  "latency_ms": 412.7
}
```

`embedding_provider` and `vector_store` are class names; `embedding_model` is currently always
`"unknown"`. The check makes a real embedding call, so it consumes provider quota.

### `POST /rag/embed`

| Field | Type | Rules |
|---|---|---|
| `text` | string | 1–100,000 chars, not blank |

```json
{ "embedding": [0.0123, -0.0456, "..."], "dimension": 1536, "latency_ms": 0 }
```

`latency_ms` is currently always `0` for this endpoint.

### `POST /rag/ingest`

Requires a non-guest identity.

| Field | Type | Default | Rules |
|---|---|---|---|
| `documents` | list | — | 1–100 items |
| `documents[].doc_id` | string | — | not blank |
| `documents[].content` | string | — | not blank, ≤ 1,000,000 chars |
| `documents[].metadata` | object \| null | `null` | copied onto every chunk |
| `documents[].access_level` | `public` \| `internal` \| `restricted` \| `confidential` | `internal` | who may retrieve the document |
| `documents[].allowed_roles` | list of roles | `[]` | roles allowed to read a `restricted` document |
| `documents[].organization_id` | string \| null | caller's `X-Organization-Id` | owning tenant; non-admins may only use their own |

The access fields are stored in each chunk's metadata in the vector store, so they are
enforced on every search, including after restarts. Chunks ingested before these fields existed
are treated as `internal` with no organization (visible to any non-guest in any tenant);
re-ingest them to restrict them.

```json
{ "chunks_created": 12, "documents_stored": 2, "latency_ms": 1830.4 }
```

### `POST /rag/search`

| Field / parameter | Type | Default | Rules |
|---|---|---|---|
| `query` | string | — | 1–10,000 chars, not blank |
| `top_k` | int | `5` | 1–100 |
| `similarity_threshold` | float \| null | `null` | 0.0–1.0 |
| `?rerank=` | bool | `false` | rerank with the cross-encoder |
| `?enhance=` | bool | `false` | also search LLM query expansions (needs an OpenAI key) |

Pipeline:

1. **Guardrails** — unsafe queries return `400` before any search.
2. **Enhancement** (`?enhance=true`) — LLM expansions are re-validated by the guardrails;
   unsafe ones are dropped and logged. Each remaining query is searched; hits are merged by
   `chunk_id`, keeping the best score.
3. **Vector search** — fetches `4 × top_k` candidates (max 100), because later stages drop hits.
4. **Access control** — hits the caller may not read are removed.
5. **Reranking** (`?rerank=true`) — cross-encoder scores the remaining hits; results are
   ordered by `rerank_score`.
6. **Cut to `top_k`, then PII masking** of each passage. If Presidio cannot load, the request
   fails with `503` rather than returning unmasked text.

```json
{
  "results": [
    {
      "chunk_id": "sec-101#0",
      "text": "Rotate encryption keys every 90 days. Questions: [REDACTED]",
      "similarity_score": 0.83,
      "metadata": {
        "chunk_id": "sec-101#0",
        "document_id": "sec-101",
        "doc_id": "sec-101",
        "sequence": 0,
        "metadata": { "team": "security", "access_level": "internal", "allowed_roles": [], "organization_id": "acme" }
      },
      "rerank_score": null
    }
  ],
  "latency_ms": 95.2
}
```

The raw chunk text stored in the payload is removed from `metadata`; only the masked `text` is
returned. Values inside `metadata.metadata` (your own ingest metadata) are **not** PII-masked.

### `POST /rag/query`

Same pipeline and parameters as `/rag/search`, without `similarity_threshold`. Returns context
passages only; answer generation happens in the agent layer.

### `GET /rag/collections`

```json
{
  "collections": [
    { "name": "documents", "vector_size": 1536, "count": 240, "distance": "cosine" }
  ]
}
```

### `POST /rag/verify`

Checks an LLM answer against the passages it was given. Runs offline: no provider call, no
identity required.

| Field | Type | Default | Rules |
|---|---|---|---|
| `query` | string | — | 1–10,000 chars |
| `answer` | string | — | 1–100,000 chars, not blank |
| `sources` | list | `[]` | ≤ 100 passages |
| `sources[].doc_id` | string | — | not empty |
| `sources[].text` | string | — | passage given to the LLM |
| `sources[].title` | string \| null | `doc_id` | |
| `sources[].score` | float | `1.0` | retrieval score, 0.0–1.0 |

```json
{
  "citations": [
    { "claim": "Encryption keys rotate every 90 days", "doc_ids": ["sec-101"], "confidence": 0.7 }
  ],
  "confidence": 0.65,
  "risk_level": "low",
  "risk_score": 0.0,
  "patterns_detected": [],
  "explanation": "No hallucination patterns detected.",
  "latency_ms": 1.8
}
```

With no `sources`, the answer is reported as `high` risk with `unsupported_claim`. See
[section 2.6](#26-hallucination-detection--hallucination_detector) for how risk is computed.

### Examples

```powershell
uv run uvicorn aegis_platform.api.main:app --reload
```

```bash
curl -X POST http://localhost:8000/rag/health

curl -X POST http://localhost:8000/rag/ingest -H "Content-Type: application/json" \
  -H "X-User-Id: alice" -H "X-User-Role: analyst" -H "X-Organization-Id: acme" -d '{
  "documents": [
    {"doc_id": "sec-101", "content": "Rotate encryption keys every 90 days.", "metadata": {"team": "security"}},
    {"doc_id": "sec-102", "content": "Access reviews run quarterly.", "access_level": "restricted", "allowed_roles": ["analyst"]}
  ]
}'

curl -X POST "http://localhost:8000/rag/search?rerank=true" -H "Content-Type: application/json" \
  -H "X-User-Id: alice" -H "X-User-Role: analyst" -H "X-Organization-Id: acme" \
  -d '{"query": "How often should keys be rotated?", "top_k": 3, "similarity_threshold": 0.5}'

curl -X POST http://localhost:8000/rag/verify -H "Content-Type: application/json" -d '{
  "query": "How often should keys be rotated?",
  "answer": "Rotate encryption keys every 90 days.",
  "sources": [{"doc_id": "sec-101", "text": "Rotate encryption keys every 90 days."}]
}'

curl http://localhost:8000/rag/collections
```

```python
import httpx

headers = {"X-User-Id": "alice", "X-User-Role": "analyst", "X-Organization-Id": "acme"}
with httpx.Client(base_url="http://localhost:8000", headers=headers, timeout=30) as client:
    resp = client.post(
        "/rag/query",
        params={"rerank": "true"},
        json={"query": "How often should keys be rotated?", "top_k": 3},
    )
    if resp.status_code == 400:
        print("rejected:", resp.json()["detail"])
    resp.raise_for_status()
    for hit in resp.json()["results"]:
        print(f"{hit['rerank_score']:.2f}  {hit['text']}")
```

---

## 2. Python API — safety modules

All modules live in `aegis_platform.rag`. Each exposes a module-level singleton accessor whose
`config` argument is **only applied on the first call**; later calls return the existing
instance unchanged. Construct the class directly when you need a differently configured
instance.

| Layer | Module | Accessor | Sync/async |
|---|---|---|---|
| Query guardrails | `query_guardrails` | `get_query_guardrails(config=None)` | sync |
| Query enhancement | `llm_query_enhancer` | `get_llm_query_enhancer(config=None)` | async |
| Access control (RBAC) | `access_control` | `DocumentAccessControl.get_instance(config=None)` | mixed |
| Reranking | `cross_encoder_reranker` | `get_cross_encoder_reranker(config=None)` | async |
| Citation tracking | `citation_tracker` | `get_citation_tracker()` | async |
| Hallucination detection | `hallucination_detector` | `get_hallucination_detector()` | async |
| PII detection/masking | `pii_detector` | `get_pii_detector(config=None)` | sync, plus async wrappers |
| Pipeline used by the API | `secure_retrieval` | `secure_retrieve(...)`, `verify_answer(...)` | async |

### 2.1 Query guardrails — `query_guardrails`

```python
QueryGuardrails(config: GuardrailsConfig | None = None)
.validate_query(query: str) -> GuardResult      # raises ValueError if query is empty
.sanitize_query(query: str) -> str              # best-effort cleanup; validate the result again
```

`GuardResult` attributes: `is_safe`, `blocked` (`not is_safe`), `query`, `violations: list[str]`,
`risk_level` — `"LOW"` or `"HIGH"` (**uppercase**).

Every failed check sets `risk_level="HIGH"`. SQL checks match statement shapes, not bare
keywords, so `How do I select a model?`, `C# best practices` and `How do I delete my account?`
pass.

| Check | Matches | Example violation |
|---|---|---|
| Length > `max_query_length` | | `Query exceeds maximum length (50000 > 10000)` |
| Words × 1.3 > `max_query_tokens` | | `Query exceeds maximum tokens (13000 > 2000)` |
| SQL statements | `UNION SELECT`, `SELECT *`, `SELECT a, b FROM`, `INSERT INTO`, `UPDATE t SET`, `DELETE FROM`, `DROP TABLE`, `TRUNCATE TABLE`, `ALTER TABLE`, `CREATE TABLE x (`, `EXEC xp_…`, `information_schema`, `WAITFOR DELAY`, `SLEEP(5)`, `CHAR(65)` | `Detected SQL keyword: DROP` |
| SQL comments | `--` or `#` right after a quote, `;` or `)`; `/* … */` | `Detected SQL comment patterns` |
| SQL tautologies | `' OR '1'='1`, `' or 1=1`, `OR 1=1` | `Detected SQL tautology pattern` |
| Prompt injection | `ignore … previous`, `system:`, `[ADMIN]`, `{{…}}`, … | `Detected prompt injection pattern: ...` |
| Command injection | `;`, `&&`, `\|`, `` ` `` or `$(` followed by a shell command (`rm`, `curl`, `sh`, …); `$(…)`; `rm -rf`; `/etc/passwd`; `../../`; `> /dev/tcp/` | `Detected command injection pattern: destructive rm flags` |
| Control characters or null bytes | | `Query contains control characters` |

Pattern matching reduces risk but is not exhaustive; treat it as one layer of defence.

### 2.2 Query enhancement — `llm_query_enhancer`

```python
LLMQueryEnhancer(config: LLMQueryEnhancerConfig | None = None)
await .enhance(query: str, strategy: EnhancementStrategy | None = None) -> EnhancedQuery
await .expand_query(query: str) -> list[str]
await .rewrite_query(query: str) -> str
await .batch_enhance(queries: list[str], strategy=None) -> list[EnhancedQuery]
reset_llm_query_enhancer_instance() -> None
```

`EnhancementStrategy`: `EXPANSION`, `REWRITE`, `HYBRID` (both, run in parallel).

`EnhancedQuery` fields: `original_query`, `expanded_queries`, `rewritten_query`,
`expansion_terms`, `strategy_used`, `confidence_score`, `metadata`.

With `include_original=True` (default) the normalized (lower-cased) query is the first item of
`expanded_queries`. If the LLM call fails, expansion returns only the normalized query and
rewriting returns the normalized query: enhancement degrades instead of raising.

### 2.3 Access control — `access_control`

```python
DocumentAccessControl(config: AccessControlConfig | None = None)
.register_document(doc: DocumentMetadata) -> None
await .check_access(user: UserContext, doc_id: str, operation: str = "read") -> AccessCheckResult
await .check_access_batch(user, doc_ids: list[str]) -> dict[str, AccessCheckResult]
.evaluate(user, doc: DocumentMetadata, operation: str = "read") -> AccessCheckResult   # no registry, no audit
.filter_search_results(user, search_results: list[dict]) -> list[dict]   # reads "doc_id" or "id"
await .get_accessible_documents(user, limit: int = 100) -> list[DocumentMetadata]
DocumentAccessControl.get_instance(config=None)
DocumentAccessControl.reset_instance()
```

Models:

- `Role`: `ADMIN`, `ANALYST`, `RESEARCHER`, `USER`, `GUEST`
- `AccessLevel`: `PUBLIC`, `INTERNAL`, `RESTRICTED`, `CONFIDENTIAL`
- `UserContext(user_id, role, organization_id=None, permissions=[])`
- `DocumentMetadata(doc_id, title, access_level=INTERNAL, allowed_roles=[], owner=None, organization_id=None, tags=[])`
- `AccessCheckResult(is_allowed, reason, risk_level, audit_log_entry)` — `risk_level` is lowercase

`check_access` decides in this order:

1. Access control disabled → allow.
2. Document not registered → deny (`fail_secure=True`, default) or raise `ValueError`.
3. User and document both have an `organization_id` and they differ → deny (`medium`).
4. `ADMIN` → allow.
5. `PUBLIC` document → allow.
6. `GUEST` → deny.
7. `INTERNAL` document → allow.
8. `RESTRICTED` document → allow if the role is in `allowed_roles` or inherits one of them via
   `role_hierarchy` (e.g. `ANALYST` inherits `USER`); otherwise deny (`medium`).
9. `CONFIDENTIAL` document → deny (`high`).

Documents and users without an `organization_id` skip the tenant check, so always set it in
multi-tenant deployments.

`evaluate(user, doc)` applies steps 1 and 3–9 to metadata you pass in, without the registry
and without audit logging. `filter_search_results` and `get_accessible_documents` use it for
registered documents; the `/rag` API uses it with metadata read from the vector store.

When `audit_enabled`, each `check_access` call fills `result.audit_log_entry` (`timestamp`,
`user_id`, `user_role`, `organization_id`, `operation`, `resource_type`, `resource_id`,
`action`, `result`, `reason`, `risk_level`) and logs it. `filter_search_results` logs one summary
line, not per-document entries.

### 2.4 Reranking — `cross_encoder_reranker`

```python
CrossEncoderReranker(config: CrossEncoderConfig | None = None)
await .rerank(query: str, results: list[RawResult], top_k: int | None = None) -> list[RankedResult]
await .score(query: str, document: str) -> float
await .batch_score(query: str, documents: list[str]) -> list[float]
.reset_model() -> None
reset_cross_encoder_instance() -> None
```

- `RawResult(result_id, text, similarity_score=None, source_id=None, metadata={})`
- `RankedResult(result_id, text, relevance_score, source_id, metadata, original_rank, new_rank)` — ranks are 1-based

The model loads on the first scoring call. Models that output raw logits (including the default
`cross-encoder/ms-marco-MiniLM-L-6-v2`) are mapped to 0–1 with a sigmoid; models that already
apply an activation are clamped to 0–1. Results are sorted by `relevance_score`, filtered by
`score_threshold`, then cut to `top_k`.

### 2.5 Citation tracking — `citation_tracker`

```python
CitationTracker()
await .track_retrieval_context(query, retrieved_docs: list[DocumentMetadata],
                               retrieval_scores: list[float], passages: list[str]) -> SourceContext
await .extract_citations(llm_response: str, source_context: SourceContext) -> list[Citation]
await .calculate_response_confidence(citations, source_context) -> float
await .verify_citations(citations, source_context) -> VerificationResult
```

- `track_retrieval_context` raises `ValueError` if the three lists differ in length.
- `extract_citations` splits the response on `.` and cites a passage when more than 30% of the
  non-stop-words overlap. It raises `ValueError` when the response is empty **or the context has
  no passages**.
- `Citation(claim, sources, confidence)`;
  `CitationSource(doc_id, title, source_relevance_score, passage_start, passage_end, passage_text)`.
- Response confidence = 0.5 × mean citation confidence + 0.3 × coverage + 0.2 × source diversity.
- `VerificationResult` keys: `total_citations`, `verified_citations`, `unverified_citations`,
  `contradicted_citations`, `details`.

### 2.6 Hallucination detection — `hallucination_detector`

```python
HallucinationDetector()
await .detect_hallucinations(llm_response: str, source_context: SourceContext,
                             citations: list[Citation]) -> HallucinationScore
```

`HallucinationScore(patterns_detected, risk_level, risk_score, explanation)` — `risk_level` is
`"low"`, `"medium"` or `"high"` (**lowercase**, unlike guardrails).

| Pattern | Raised when |
|---|---|
| `UNSUPPORTED_CLAIM` | No citations, or cited claims cover < 50% of the response's words |
| `CONTRADICTION` | Response negates ("is not", "is false", "denies") while a source asserts |
| `NUMERICAL_FABRICATION` | A number > 100 in the response is not within 10% of any source number |
| `OUT_OF_CONTEXT` | < 30% of the response's words longer than 4 chars appear in the sources |
| `ATTRIBUTION_ERROR`, `INVENTED_EVIDENCE` | Defined, **not detected yet** |

Risk score: `1.0` with no citations; `0.0` with citations and no patterns; otherwise
0.2 × number of patterns + 0.3 × (1 − mean citation confidence), capped at 1.0.
Levels: < 0.33 low, < 0.66 medium, otherwise high.

### 2.7 PII detection and masking — `pii_detector`

```python
PIIDetector(config: PIIConfig | None = None)    # raises ImportError without Presidio
.detect_pii(text: str) -> list[PII]             # raises ValueError on empty text
.mask_pii(text: str) -> str                     # raises ValueError on empty text
.mask_pii_in_dict(data: dict, keys_to_mask: list[str] | None = None) -> dict
.get_supported_entities() -> list[str]
await .async_detect_pii(text) / await .async_mask_pii(text)
```

`PII` attributes: `entity_type`, `text`, `start`, `end`, `confidence`.

Entities: `EMAIL_ADDRESS`, `PHONE_NUMBER`, `US_SSN`, `CREDIT_CARD`, `US_BANK_ACCOUNT`,
`US_PASSPORT`, `US_DRIVER_LICENSE`, `IP_ADDRESS`; `PERSON` with `detect_names=True`;
`LOCATION` with `detect_addresses=True`.

`detect_pii` and `mask_pii` both act only on entities scoring at least `confidence_threshold`,
and every masked entity is replaced with `mask_char` (default `[REDACTED]`).

**Masking fails closed.** If Presidio raises inside `mask_pii` or `mask_pii_in_dict`, a
`RuntimeError("PII masking failed: ...")` is raised; unmasked text is never returned.
`mask_pii_in_dict` leaves empty strings and non-string values unchanged.

### 2.8 Safety pipeline — `secure_retrieval`

The code behind `/rag/search`, `/rag/query` and `/rag/verify`, usable directly:

```python
await secure_retrieve(service: RAGService, query: str, user: UserContext, top_k: int = 5,
                      similarity_threshold: float | None = None,
                      rerank: bool = False, enhance: bool = False) -> dict   # {"results", "latency_ms"}
await verify_answer(query: str, answer: str,
                    sources: list[tuple[doc_id, title, text, score]]) -> dict
access_metadata(access_level, allowed_roles, organization_id) -> dict      # fields stored at ingest
document_metadata_from_payload(payload: dict) -> DocumentMetadata          # rebuild for access checks
class QueryRejectedError(ValueError)                                       # .violations, .risk_level
```

`secure_retrieve` raises `QueryRejectedError` for unsafe queries and `RuntimeError` if search,
reranking or PII masking fails. `document_metadata_from_payload` treats missing access fields as
`internal` and unknown access levels as `confidential`, so a malformed payload never widens
access.
