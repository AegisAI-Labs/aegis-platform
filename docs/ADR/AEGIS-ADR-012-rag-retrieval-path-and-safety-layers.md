# ADR-012: RAG Retrieval Path Contract and Safety Layers

- **Status:** Accepted
- **Date:** 2026-11-20 (Day 16)
- **Related:** RFC-009, ADR-009 (chunking, embedding and vector store selection), ADR-010 (service wiring),
  ADR-007 (single retry layer), SRS FR-017 / FR-018

## Context

The RAG safety layers (Day 12, commits `3dbdb79`, `1105c66`, `51dc626`) shipped without an RFC, ADR or SRS
entry. The Day 13 assessment also found that the real embed → Qdrant path could not work outside tests:

- `OpenAIEmbedding.embed_text` took an `EmbeddingRequest` and returned an `EmbeddingResponse`, while
  `RAGService` calls `embed_text(str) -> np.ndarray` (the `EmbeddingProvider` protocol). A
  `# type: ignore` in the factory hid the mismatch; the API tests replaced `RAGService` with a fake.
- `QdrantVectorStore.search` called `AsyncQdrantClient.search`, which qdrant-client 1.20 does not have.
- Point IDs were `hash(id) % 2**31`; Python randomizes string hashing per process, so IDs changed on restart
  (duplicates on re-ingest, deletes/updates missed their points) and could collide.
- The `documents` collection was never created; `create_collection` used `recreate_collection`, which
  deletes existing data.
- The vector store was built with a fixed 1536 dimension regardless of the embedding model.
- `/rag/collections` always failed validation (names returned where `CollectionInfo` objects are required);
  `/rag/health` reported healthy even when Qdrant was down.
- Live testing on Day 16 showed NVIDIA's asymmetric Nemotron embedding model rejects requests without
  `input_type`, and both embedders retried errors that can never succeed (HTTP 400, quota exhausted).
- Tenant isolation was skipped when a caller sent no `X-Organization-Id`.

## Decision

### Retrieval path contract

1. Every embedder implements `EmbeddingProvider` exactly: `embed_text(str) -> np.ndarray` (used for queries
   and health checks) and `embed_batch(list[str]) -> list[np.ndarray]` (used for ingested passages).
   Output length and order always match the input; empty texts are rejected, never dropped. Large inputs
   are split into API-sized batches.
2. Asymmetric embedding models receive `input_type`: `"query"` from `embed_text`, `"passage"` from
   `embed_batch` (`RAG_EMBEDDING_SEND_INPUT_TYPE=false` disables it for symmetric models).
3. Embedders retry only transient failures (rate limit, 5xx, connection errors) with bounded backoff;
   client errors and quota exhaustion fail immediately. SDK-internal retries are disabled. Error messages
   contain status codes or exception types only, never response bodies or URLs.
4. Qdrant point IDs: numeric strings stay integers; every other ID maps to a deterministic UUIDv5 in a
   fixed namespace. Re-ingesting a chunk overwrites it. The namespace is a code constant and is **not**
   configurable: changing it would make existing points unreachable by chunk ID (deletes and updates
   would silently miss them), and separate collections already provide separate ID spaces.
5. **Re-ingesting a document replaces it.** After the new chunks are upserted, the document's other points
   (same payload `document_id`, ID not in the new chunk set) are deleted via
   `VectorStore.delete_stale_chunks`. Upsert-then-delete means a failed ingest leaves the previous version
   searchable, and a shorter new version leaves no outdated tail chunks. The same step removes points
   written with pre-Day-16 hash IDs. `document_id` has a keyword payload index so the cleanup is a filtered
   lookup. Ingest responses report `chunks_removed`.
6. **One collection per embedding model.** The collection is configurable (`RAG_COLLECTION`, default
   `documents`). Vectors from different models are not comparable even at the same dimension, so a new
   collection records `embedding_model`, `embedding_dimension` and `distance_metric` in its Qdrant
   metadata. Search, ingest, cleanup and `/rag/health` verify an existing collection against the current
   configuration and refuse a mismatch with `CollectionMismatchError` ("set RAG_COLLECTION to a new
   collection and re-ingest"). Collections without recorded metadata (created before this check) are
   verified by dimension and metric only.
7. Collections are created on first insert with the embedding provider's dimension and the configured
   distance metric. `create_collection` refuses to overwrite an existing collection. Searching a collection
   that does not exist yet returns no results.
8. Search uses `query_points`. Health checks fail when either the embedder or the vector store is down.
9. `RAG_VECTOR_STORE_URL=:memory:` runs qdrant-client in-process. The test suite uses it so the real client
   API is exercised without Docker; it is not a production setting.

### Safety layers (retroactive record of Day 12 behavior)

`/rag/search` and `/rag/query` run, in order (`rag/secure_retrieval.py`):

| Step | Behavior | Known limits |
|---|---|---|
| Query guardrails | Length limits; regex checks for SQL, shell and prompt injection | Pattern lists are short and bypassable; not a complete defense |
| Optional query enhancement | LLM expansion; every expansion is re-checked by the guardrails | Bypasses the platform `LLMProvider` chain (Day 17) |
| Vector search | Over-fetches 4 × `top_k` (max 100) to leave room for filtering | |
| Access control | Per-chunk `access_level`, `allowed_roles`, `organization_id` stored in the payload; unknown levels treated as CONFIDENTIAL | Identity comes from trusted gateway headers |
| Optional rerank | Cross-encoder `ms-marco-MiniLM-L-6-v2` | Runs synchronously on the event loop (Day 17) |
| PII masking | Presidio (`en_core_web_lg`) on returned text; fails closed if masking is unavailable; raw `text` removed from returned metadata | Raw text is stored unmasked in Qdrant |

`/rag/verify` scores citations and hallucination risk with word-overlap heuristics (no NLI or LLM judge).

**Tenant rule (Day 16):** a document with an `organization_id` is readable only by callers from the same
organization, unless the document is PUBLIC. This now also applies when the caller sends no
`X-Organization-Id` (previously the check was skipped).

## Consequences

### Positive

- The default configuration works end to end against a real Qdrant client.
- Restarts and re-ingestion no longer create duplicate, outdated or orphaned points.
- Switching embedding models can no longer silently mix incompatible vectors in one collection.
- The embedder/store contract is enforced by tests that use the real client, not mocks of it.
- Callers without an organization can no longer read other tenants' internal documents.

### Trade-offs / compatibility

- Points written before Day 16 used random hash IDs; re-ingesting a document now removes them. Documents
  that are never re-ingested keep their old points until the collection is recreated.
- Each ingest makes one count and at most one delete call per document.
- Not handled: identical text stored under different document IDs (needs content hashing).
- `OpenAIEmbedding.embed_text` no longer accepts `EmbeddingRequest`; callers pass a string.
- Requests without `X-Organization-Id` see fewer results for organization-scoped documents.
- Live validation on Day 16 was blocked by provider availability (NVIDIA 503 "overloaded", OpenAI account
  without credits); see EDI Day 16.
