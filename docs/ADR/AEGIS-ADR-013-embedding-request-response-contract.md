# ADR-013: Embedding Request/Response Contract with Usage

- **Status:** Accepted
- **Date:** 2026-11-21 (Day 16)
- **Related:** ADR-004 (LLM provider abstraction), ADR-008 (observability), ADR-012 (retrieval path)

## Context

The LLM layer returns `LLMResponse` with `LLMUsage`, which observability already reads. The embedding layer
returned bare vectors (`embed_text -> np.ndarray`, `embed_batch -> list[np.ndarray]`), so token usage and the
model that served the call were discarded, although both OpenAI and NVIDIA report them (verified live on
Day 16: NVIDIA returns `usage.prompt_tokens` and `model`). Usage is needed later for embedding metrics
(roadmap W9) and cost accounting (AI-economics workstream).

`EmbeddingRequest` / `EmbeddingResponse` already existed in `rag/models.py` from the Day 9 design but were
unused, and their shape was wrong for this purpose:

- one text per request, so batches needed one object per text;
- `tokens_used` per text, while providers report usage per API call (the old OpenAI code copied the batch
  total onto every item, overcounting by the batch size);
- no way to say whether text is a search query or a stored passage (required by asymmetric models).

## Decision

1. Redefine the models as batch-level, mirroring the LLM layer:
   - `EmbeddingRequest(texts, input_type="passage" | "query")`
   - `EmbeddingUsage(input_tokens: int | None)`
   - `EmbeddingResponse(embeddings, model, provider, dimension, usage: EmbeddingUsage | None)`
2. `EmbeddingProvider.embed(request) -> EmbeddingResponse` is the primary method. `embed_text` (query) and
   `embed_batch` (passages) remain as thin wrappers, so existing callers are unchanged.
3. Usage is summed across the API calls a request is split into. If any call does not report usage, the
   total is `None` (unknown), never a partial number.
4. `RAGService` calls `embed()` and returns `embedding_tokens` from ingestion and search. No API schema,
   metric or cost calculation changes in this step; exporting usage belongs to W9 and the AI-economics
   workstream.

## Consequences

### Positive

- Embedding token usage and serving model are available to callers instead of being discarded.
- The query/passage distinction is explicit in the contract rather than implied by which method is called.
- One request/response shape across LLM and embedding providers.

### Trade-offs / compatibility

- The unused Day 9 `EmbeddingRequest(text, model)` / `EmbeddingResponse(text, embedding, ..., tokens_used)`
  shapes are replaced. Nothing in `src`, `tests` or `scripts` used them.
- Third-party `EmbeddingProvider` implementations must add `embed()`.
