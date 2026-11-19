# RAG Safety Configuration Guide

**Scope:** TASK-048 safety layers and the `/rag` API
**Related:** [API reference](../API/AEGIS-API-RAG.md) · [Usage guide](RAG-SAFETY-USAGE.md)

This guide covers installation, environment variables, per-module settings with their defaults,
recommended production values, and troubleshooting.

---

## 1. Installation

All safety dependencies are declared in `pyproject.toml`, including the spaCy model Presidio needs.

```powershell
uv sync
```

| Package | Used by | Notes |
|---|---|---|
| `presidio-analyzer`, `presidio-anonymizer` | PII detection/masking | Pulls in spaCy |
| `en-core-web-lg` | Presidio's default NLP engine | ~400 MB wheel, pinned via `[tool.uv.sources]` |
| `sentence-transformers` | Cross-encoder reranking | Pulls in PyTorch (CPU build on Windows) |
| `openai` | Query enhancement, embeddings | Needs `OPENAI_API_KEY` |

The reranker model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~90 MB) downloads from Hugging Face
on first use and is cached in `~/.cache/huggingface`. Pre-fetch it on build machines:

```powershell
uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

Verify everything loads:

```powershell
uv run python scripts/rag_safety_pipeline_example.py
```

Each of the seven layers should print results; a layer whose dependency is missing prints
`skipped: ...` with the reason.

---

## 2. Environment variables

### 2.1 Loading `.env`

The API calls `load_dotenv()` at start-up (`api/main.py`), exporting `.env` from the working
directory into `os.environ`; variables already set in the environment win. Start the server
from the repository root:

```powershell
uv run uvicorn aegis_platform.api.main:app --reload
```

Scripts that import the `aegis_platform.rag` modules directly (without the API) do not load
`.env`; use `uv run --env-file .env python <script>` for those.

### 2.2 Variables read by the `/rag` API

Where two names are listed, the first non-empty one wins; the `QDRANT_*` / `OPENAI_API_KEY`
names are accepted for backward compatibility.

| Variable | Default | Notes |
|---|---|---|
| `RAG_EMBEDDING_PROVIDER` | `openai` | `openai` or `nvidia`; anything else returns `503` |
| `RAG_EMBEDDING_KEY`, then `OPENAI_API_KEY` | — | Embedding key for `openai` |
| `RAG_EMBEDDING_KEY` | — | Required for `nvidia` |
| `RAG_EMBEDDING_MODEL`, `RAG_EMBEDDING_DIMENSION`, `RAG_EMBEDDING_BATCH_SIZE`, `RAG_EMBEDDING_TIMEOUT`, `RAG_EMBEDDING_MAX_RETRIES`, `RAG_EMBEDDING_API_ENDPOINT` | see `NVIDIAEmbeddingConfig` | Used by `nvidia` only; `openai` always uses `text-embedding-3-small` (1536 dimensions) |
| `RAG_VECTOR_STORE` | `qdrant` | Only `qdrant` is supported |
| `RAG_VECTOR_STORE_URL`, then `QDRANT_URL` | `http://localhost:6333` | |
| `RAG_VECTOR_STORE_API_KEY`, then `QDRANT_API_KEY` | unset | |
| `RAG_VECTOR_STORE_DISTANCE_METRIC` | `cosine` | `cosine`, `euclidean` or `manhattan` |
| `OPENAI_API_KEY` | — | Also used by query enhancement (`?enhance=true`) through the OpenAI SDK |

Chunking variables (`RAG_CHUNKING_STRATEGY`, `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`) are documented
in the [RAG module README](../../src/aegis_platform/rag/README.md#configuration-via-environment-variables).

### 2.3 Safety modules

The safety modules have **no environment variables**. They are configured in code by passing a
config object to the constructor or to the singleton accessor's first call (see section 3).

### 2.4 Identity headers

`/rag/search`, `/rag/query` and `/rag/ingest` read the caller from `X-User-Id`, `X-User-Role` and
`X-Organization-Id`. The API trusts these headers, so it must run behind a gateway that
authenticates users and overwrites the headers. See the
[API reference](../API/AEGIS-API-RAG.md#caller-identity).

---

## 3. Module configuration

Singleton accessors (`get_query_guardrails(config)`, `get_pii_detector(config)`, …) apply `config`
only when they create the instance. Configure them once at application start-up, before any
request handling, or construct the classes directly.

### 3.1 Query guardrails — `GuardrailsConfig`

| Field | Default | Effect |
|---|---|---|
| `enabled` | `True` | `False` returns `is_safe=True` for every non-empty query |
| `max_query_length` | `10000` | Characters; longer queries are blocked |
| `max_query_tokens` | `2000` | Estimated as words × 1.3 |
| `block_sql_patterns` | `True` | SQL statement shapes, comment markers after a quote/`;`/`)`, tautologies |
| `block_prompt_patterns` | `True` | Prompt-injection regexes |
| `block_command_patterns` | `True` | Shell command injection (`; rm -rf`, `\| sh`, `$(…)`, `/etc/passwd`, `../../`) |

### 3.2 Query enhancement — `LLMQueryEnhancerConfig`

| Field | Default | Effect |
|---|---|---|
| `strategy` | `EXPANSION` | `EXPANSION`, `REWRITE` or `HYBRID` |
| `llm_provider` | `openai` | `openai` or `anthropic` (`anthropic` package not installed by default) |
| `model_name` | `gpt-3.5-turbo` | Chat model used for expansion/rewrite |
| `temperature` / `top_p` | `0.7` / `0.9` | Lower temperature gives more repeatable expansions |
| `max_tokens` | `256` | Per LLM call |
| `max_expansions` | `5` | Terms kept from the LLM output |
| `include_original` | `True` | Prepend the normalized query to `expanded_queries` |
| `normalize_query` | `True` | Lower-case and strip before caching and expansion |
| `timeout_seconds` | `5.0` | Per LLM call |
| `cache_enabled` | `True` | In-memory, per instance, no eviction |
| `batch_size` | `5` | Concurrency for `batch_enhance` |

`enable_caching` and `cache_ttl_seconds` exist on the config but are not used; caching is
controlled by `cache_enabled` only, and entries never expire.

### 3.3 Access control — `AccessControlConfig`

| Field | Default | Effect |
|---|---|---|
| `enabled` | `True` | `False` allows everything (`check_access`, `evaluate`, `filter_search_results`, the API) |
| `fail_secure` | `True` | Unknown documents are denied (and dropped from search results); `False` raises / passes them through |
| `audit_enabled` | `True` | Attach and log an audit entry for every `check_access` |
| `audit_log_level` | `info` | Level used for audit log lines |
| `role_hierarchy` | admin ⊃ analyst/researcher ⊃ user ⊃ guest | Inheritance used for `RESTRICTED` documents |
| `default_access_level` | `INTERNAL` | Stored but not used; `DocumentMetadata` itself defaults to `INTERNAL` |

The document registry is in memory. Register every document's metadata at ingestion time
(and after restarts), otherwise `fail_secure` denies it.

### 3.4 Reranking — `CrossEncoderConfig`

| Field | Default | Effect |
|---|---|---|
| `model_name` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Any Hugging Face cross-encoder |
| `device` | `cpu` | `cuda` or `mps` when available |
| `max_length` | `512` | Tokens per query+document pair; longer pairs are truncated |
| `batch_size` | `32` | Pairs per inference batch |
| `top_k` | `10` | Results returned when `rerank(top_k=None)` |
| `score_threshold` | `0.0` | Drop results scoring below this (0–1) |
| `normalize_scores` | `True` | Sigmoid for logit models, clamp otherwise |
| `return_metadata` | `True` | Copy `RawResult.metadata` onto results |
| `trust_remote_code` | `False` | Keep `False` unless you have reviewed the model's code |

### 3.5 PII detection — `PIIConfig`

| Field | Default | Effect |
|---|---|---|
| `enabled` | `True` | Stored; not currently checked by `detect_pii`/`mask_pii` |
| `mask_char` | `[REDACTED]` | Replacement for every masked entity |
| `confidence_threshold` | `0.4` | Minimum Presidio score to detect or mask |
| `detect_names` | `False` | Add `PERSON` |
| `detect_addresses` | `False` | Add `LOCATION` |

Typical Presidio scores: email, credit card `1.0`; US SSN `0.85`; phone number `0.4`. A threshold
above `0.4` stops phone numbers from being detected or masked.

Masking fails closed: a Presidio error raises `RuntimeError` instead of returning unmasked text,
and the `/rag` API answers `503`.

### 3.6 Citation tracking and hallucination detection

`CitationTracker` and `HallucinationDetector` take no configuration. Their thresholds (30% word
overlap for a citation, 50% citation coverage, 30% topic overlap, numbers > 100 within 10%) are
fixed in code; see the [API reference](../API/AEGIS-API-RAG.md#26-hallucination-detection--hallucination_detector).

---

## 4. Recommendations

### Security-first defaults

| Setting | Recommendation | Why |
|---|---|---|
| Identity headers | Set `X-User-*` / `X-Organization-Id` only at an authenticating gateway; strip client copies | The API trusts them as-is |
| Guardrails | Keep all three pattern families on; run on the user query **and** on every LLM expansion (the API does both) | LLM output is untrusted input |
| `AccessControlConfig.fail_secure` | `True` | Unregistered documents must never leak |
| `organization_id` | Set on every user and document (the API defaults it from the caller) | The tenant check is skipped when either side is `None` |
| Legacy chunks | Re-ingest data ingested before access metadata existed | It defaults to `internal` with no tenant |
| `PIIConfig.confidence_threshold` | `0.4` (default) | Catches phone numbers |
| `detect_names` | `True` where names are sensitive | Expect some false positives |
| PII masking | Mask retrieved passages and final answers before returning or logging them | The API masks passages; ingest metadata is not masked |
| `trust_remote_code` | `False` | Avoid executing model-repo code |

### Performance

| Layer | Typical cost | Tuning |
|---|---|---|
| Guardrails | < 0.1 ms | — |
| Access control | < 1 ms per check | Use `filter_search_results` for result lists |
| PII | Several seconds one-time spaCy load; ~5 ms per short text | Create the detector at start-up |
| Reranking (CPU, MiniLM-L-6) | One-time model load; ~60 ms for 20 short passages | Rerank 20–50 candidates, not hundreds; `device="cuda"` for volume |
| Query enhancement | One LLM round-trip per uncached query | Keep `cache_enabled`; use `EXPANSION` instead of `HYBRID` if latency matters |
| Citations + hallucination | < 5 ms | — |

Measure on your own hardware with the benchmark in the
[usage guide](RAG-SAFETY-USAGE.md#9-benchmarking).

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Every `/rag/*` call returns `503 RAG Service initialization failed` | No embedding key, or the server was started outside the directory containing `.env` | Set `RAG_EMBEDDING_KEY` (or `OPENAI_API_KEY`); start from the repository root (section 2.1) |
| `503 Unknown embedding provider` | `RAG_EMBEDDING_PROVIDER` is not `openai` or `nvidia` | Fix the value |
| `/rag/search` returns only public documents, or nothing | No `X-User-Id` / `X-User-Role` headers, so the caller is a guest | Send identity headers (via the gateway) |
| `/rag/search` returns `400 Unknown X-User-Role` | Role header is not a known role | Use `admin`, `analyst`, `researcher`, `user` or `guest` |
| `/rag/ingest` returns `403` | Guest caller, or a non-admin targeting another `organization_id` | Send identity headers; omit `organization_id` to use the caller's |
| A user in one tenant sees another tenant's old documents | Chunks ingested before access metadata existed have no tenant | Re-ingest them |
| `/rag/search` returns `503 PII masking unavailable` | Presidio or the spaCy model cannot load | `uv sync`; check the error in the logs |
| A legitimate query is rejected with `400` | A guardrail pattern matched | Read `detail.violations`; rephrase, or disable that pattern family in `GuardrailsConfig` |
| `validate_query("")` raises `ValueError` | Empty queries are rejected by exception, not by `GuardResult` | Catch `ValueError` |
| Phone numbers are not masked | `confidence_threshold` above `0.4` | Lower it to `0.4` |
| SSN `123-45-6789` is not detected | Presidio deliberately ignores well-known invalid SSNs | Use realistic test data, e.g. `536-90-4399` |
| `RuntimeError: PII masking failed` | Presidio raised during masking (fails closed) | Investigate the underlying error in the message |
| `ImportError: Presidio library required` | Presidio not installed | `uv sync` |
| `OSError: [E050] Can't find model 'en_core_web_lg'` | spaCy model missing | `uv sync` (the model is a project dependency) |
| First rerank call is slow or fails offline | Model downloads from Hugging Face on first use | Pre-fetch it (section 1) |
| Every reranked score is exactly `0.0` or `1.0` | Custom model returns logits but has a non-identity activation configured | Use a model whose activation matches its training, or set `normalize_scores=False` and rank on raw scores |
| `?enhance=true` searches only the original query | LLM call failed (no `OPENAI_API_KEY`, timeout) and enhancement fell back | Check `OPENAI_API_KEY`; look for `Query expansion failed` in logs |
| Logs show `Dropped unsafe query expansion` | The LLM produced an expansion that failed guardrails | Expected; the expansion is not searched |
| Config passed to `get_*()` has no effect | The singleton already existed | Configure at start-up, or call the module's `reset_*` function first |
| `check_access` denies with `Document not found` | Document metadata never registered (Python API only; the HTTP API reads metadata from the vector store) | Call `register_document` |
| Guardrails `risk_level` is `"HIGH"` but hallucination `risk_level` is `"high"` | Modules use different casing | Compare with `.lower()` |
