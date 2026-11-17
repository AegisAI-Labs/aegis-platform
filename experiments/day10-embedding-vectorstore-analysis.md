# Day 10: Embedding Provider & Vector Store Analysis

**Date:** 2026-11-15
**Phase:** Phase 2 RAG Implementation - Provider Selection & Architecture
**Status:** Design Complete, Implementation Ready

---

## Objective

Evaluate and select concrete embedding provider and vector store implementations for Phase 2 of the RAG module, ensuring:
- Provider-agnostic architecture maintained
- Production-grade performance and reliability
- Configuration-driven behavior
- Comprehensive testing and quality standards
- Clear migration paths for future providers

---

## Executive Summary

**Phase 2 Decisions Made:**

1. **Embedding Provider:** OpenAI text-embedding-3-small
   - Cost-effective: $0.02 per 1M tokens
   - Performance: Sub-100ms latency
   - Features: Native batching (up to 2048 embeddings/request)
   - Alternative candidates evaluated: local (sentence-transformers), Cohere, Voyage AI

2. **Vector Store:** Qdrant
   - Performance: Sub-10ms search on millions of vectors
   - Deployment: Docker, Kubernetes, Qdrant Cloud
   - Architecture: HNSW indexing + metadata support
   - Alternative candidates evaluated: Pinecone, Weaviate, Milvus, Chroma

3. **Factory Pattern:** Extended registries for multi-provider support
   - EMBEDDING_PROVIDERS registry (OpenAI; future: Claude, local)
   - VECTOR_STORES registry (Qdrant; future: Pinecone, Weaviate)
   - Runtime provider selection via configuration

---

## Phase 2 Architecture Design

### Component Interaction Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   User Documents                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
        ┌────────────────────────────┐
        │  Chunker (Phase 1)         │
        │  - Recursive Strategy      │
        │  - Markdown Strategy       │
        └────────────┬───────────────┘
                     │
        Chunks + Metadata
                     │
                     v
        ┌────────────────────────────┐
        │  EmbedderService (Phase 2) │
        │  - OpenAI API Client       │
        │  - Batching Support        │
        │  - Retry & Backoff Logic   │
        └────────────┬───────────────┘
                     │
        Dense Vectors (1536 dims)
                     │
                     v
        ┌────────────────────────────┐
        │  VectorStoreService (Ph 2) │
        │  - Qdrant Client           │
        │  - Collection Management   │
        │  - CRUD Operations         │
        └────────────┬───────────────┘
                     │
              Persistent Storage
                     │
                     v
        ┌────────────────────────────┐
        │  Qdrant VectorDB           │
        │  - HNSW Index              │
        │  - Metadata Payloads       │
        │  - <10ms Search            │
        └────────────────────────────┘
```

---

## Embedding Provider Analysis

### OpenAI text-embedding-3-small (Selected)

**Why Selected:**
1. **Cost Efficiency:** $0.02/1M tokens (lowest among OpenAI options)
2. **Batching:** Native support for 2048 embeddings/request
3. **Production Track Record:** Used in major production RAG systems
4. **Quality vs. Cost:** 1536 dimensions sufficient for semantic search
5. **Integration:** Official Python SDK with clear async support

**Technical Specifications:**
- Model: `text-embedding-3-small`
- Dimensions: 1536
- Max tokens per input: 8,191 (sufficient for chunked text)
- Batch size: Up to 2,048 embeddings per API call
- Latency: Typically <100ms per batch
- Rate limit: ~10 req/sec (standard tier; higher tiers available)

**Implementation Strategy:**
- Use `openai.OpenAI` async client
- Batch multiple chunks into single request
- Implement exponential backoff for HTTP 429 (rate limiting)
- Fail immediately on HTTP 401 (auth error)
- Mock all API calls in tests (no live calls in CI/CD)

**Quality Attributes:**
- Type safety: Full type annotations (mypy --strict)
- Error handling: Retry logic with jitter
- Testing: 12 unit + 2 integration tests
- Documentation: Docstrings + README examples

---

### Alternative Candidates Evaluated

#### Local Embeddings (sentence-transformers)

```python
Model: all-MiniLM-L6-v2
Dimensions: 384
Cost: Free (open-source)
Speed: 50-200ms per batch (CPU)
```

**Evaluation:**
- ✅ Zero external dependencies
- ✅ No rate limiting
- ✅ Complete data privacy
- ❌ Slower than cloud alternatives
- ❌ Higher memory/disk overhead
- ❌ Quality varies by model
- ❌ Not enterprise-grade

**Recommendation:** Deferred to Phase 3 as alternative for cost-sensitive deployments

#### Cohere embed-english-v3.0

```
Cost: ~$0.02/1M tokens (similar to OpenAI)
Dimensions: 1024
Batching: Native support
Community: Growing adoption
```

**Evaluation:**
- ✅ Cost-competitive
- ✅ Good quality
- ❌ Smaller ecosystem
- ❌ Less community support than OpenAI
- ❌ Less adoption in LangChain/LlamaIndex

**Recommendation:** Viable alternative for Phase 3+

#### Voyage AI voyage-large-2

```
Cost: ~$0.12/1M tokens (6x OpenAI)
Dimensions: 1024
Quality: Very high
Community: Niche adoption
```

**Evaluation:**
- ✅ Very high quality
- ❌ Expensive (not cost-effective)
- ❌ Smaller ecosystem
- ❌ Over-engineered for Phase 2

**Recommendation:** Consider only if quality requirements exceed OpenAI

---

## Vector Store Analysis

### Qdrant (Selected)

**Why Selected:**
1. **Performance:** Sub-10ms search latency on millions of vectors
2. **Deployment Flexibility:** Docker, Kubernetes, managed cloud
3. **Open Source:** Full transparency, no vendor lock-in
4. **Metadata:** First-class support for rich payloads (critical for RAG)
5. **Ecosystem:** Wide adoption in LangChain, LlamaIndex, Haystack

**Technical Specifications:**
- Engine: Rust-based, high-performance
- Indexing: HNSW (Hierarchical Navigable Small World)
- Protocol: gRPC + REST API
- Scaling: Horizontal (clustering via Raft consensus)
- Deployment: Docker, Kubernetes, Qdrant Cloud

**Implementation Strategy:**
- Use `qdrant-client` Python library
- Create collections on-demand
- Support CRUD: upsert, get, delete, search
- Store chunk metadata in payload
- Implement filtering on metadata (e.g., source document)
- Mock Qdrant in tests (use in-memory mode)

**Quality Attributes:**
- Type safety: Full type annotations (mypy --strict)
- Error handling: Connection retry, collection validation
- Testing: 12 unit + 2 integration tests (in-memory mode)
- Documentation: Docstrings + README examples

---

### Alternative Candidates Evaluated

#### Pinecone (SaaS)

```
Deployment: Cloud-only (no self-hosted)
Scaling: Automatic
Indexing: Proprietary
Cost Model: Per-query charges ($0.01-$0.10 per 1K queries)
```

**Evaluation:**
- ✅ Fully managed (zero operations)
- ✅ Automatic scaling
- ✅ Optimized for speed
- ❌ Vendor lock-in (proprietary API)
- ❌ Cost model: per-query (accumulates with scale)
- ❌ No on-premise option (compliance issues)
- ❌ Frequent API breaking changes
- ❌ Data privacy concerns (embeddings on Pinecone servers)

**Recommendation:** Viable for Phase 3+ but Qdrant preferred for cost & control

#### Weaviate

```
Engine: Golang-based
Indexing: HNSW + proprietary
Deployment: Docker, Kubernetes, managed cloud
Scaling: Horizontal (Raft)
Protocol: REST + GraphQL
```

**Evaluation:**
- ✅ Open-source option
- ✅ Good documentation
- ✅ Cloud option available
- ❌ Higher memory overhead than Qdrant
- ❌ Smaller Python community
- ❌ GraphQL adds complexity

**Recommendation:** Viable alternative but Qdrant preferred for performance

#### Milvus

```
Engine: Golang-based
Indexing: Multiple options (HNSW, IVF_PQ, etc.)
Deployment: Docker, Kubernetes, Zilliz Cloud
Scaling: Horizontal (built-in)
Cost: Infrastructure only
```

**Evaluation:**
- ✅ Excellent horizontal scaling
- ✅ Cost-effective at massive scale
- ✅ Flexible indexing
- ❌ Operational complexity (more manual tuning)
- ❌ Smaller Python ecosystem
- ❌ Steeper learning curve

**Recommendation:** Consider for Phase 3+ if scale exceeds Qdrant comfort zone

#### Chroma (In-Memory)

```
Engine: Python-based
Deployment: Embedded (no separate DB)
Scaling: Single machine only
Indexing: Brute-force or simple
```

**Evaluation:**
- ✅ Easy to set up (prototype)
- ✅ No external dependencies
- ✅ Python native
- ❌ NOT production-grade
- ❌ ~100K vector limit
- ❌ Slow search at scale
- ❌ No replication/clustering

**Recommendation:** Suitable for prototypes only, NOT for Phase 2

---

## Configuration & Secrets Management

### Environment Variables

```bash
# Embedding Provider Configuration
OPENAI_API_KEY=sk-... (required)
OPENAI_EMBEDDING_MODEL=text-embedding-3-small (default)
OPENAI_BATCH_SIZE=1024 (default)
OPENAI_TIMEOUT=30 (seconds, default)
OPENAI_MAX_RETRIES=3 (default)
OPENAI_RETRY_BACKOFF_FACTOR=2 (exponential, default)

# Vector Store Configuration
QDRANT_URL=http://localhost:6333 (default)
QDRANT_API_KEY=... (optional, for Qdrant Cloud)
QDRANT_COLLECTION_NAME=aegis_embeddings (default)
QDRANT_TIMEOUT=30 (seconds, default)
QDRANT_VECTOR_SIZE=1536 (matches OpenAI embedding)
QDRANT_DISTANCE_METRIC=cosine (default)

# RAG Chunking Configuration (Phase 1)
CHUNKING_STRATEGY=recursive (default)
CHUNK_SIZE=512 (default)
CHUNK_OVERLAP=50 (default)
```

### Secrets Handling

✅ **Best Practices Implemented:**
- All secrets from environment variables
- No hardcoded credentials in source
- Secrets NOT logged or emitted in telemetry
- Validation at instantiation time
- Clear error messages for missing credentials

---

## Error Handling & Retry Strategy

### OpenAI Error Scenarios

| Error | HTTP | Behavior | Retry |
|-------|------|----------|-------|
| Rate limit | 429 | Exponential backoff with jitter | Yes |
| Token limit | 400 | Fail immediately | No |
| Auth error | 401 | Fail immediately | No |
| Server error | 5xx | Exponential backoff | Yes |
| Timeout | N/A | Exponential backoff | Yes |
| Network error | N/A | Exponential backoff | Yes |

### Qdrant Error Scenarios

| Error | Behavior | Retry |
|-------|----------|-------|
| Collection not found | Create collection on-demand | N/A |
| Connection timeout | Exponential backoff | Yes |
| Auth error | Fail immediately | No |
| Timeout | Exponential backoff | Yes |
| Network error | Exponential backoff | Yes |

### Exponential Backoff Strategy

```python
base_delay = 1.0  # seconds
max_delay = 32.0   # seconds
jitter_factor = 0.1

def next_backoff(attempt):
    delay = base_delay * (backoff_factor ** attempt)
    delay = min(delay, max_delay)
    jitter = delay * random.uniform(-jitter_factor, jitter_factor)
    return delay + jitter
```

---

## Testing Strategy

### Test Coverage Target: 30+ Tests, >85% Code Coverage

#### OpenAI EmbeddingProvider Tests (12 unit + 2 integration)

**Unit Tests (12):**
1. Instantiation with valid credentials
2. Instantiation with missing API key
3. Instantiation with invalid model name
4. Embed single chunk (mocked response)
5. Embed batch of chunks (mocked response)
6. Batch splitting (>2048 vectors)
7. Retry logic: HTTP 429 (exponential backoff)
8. Retry logic: Connection error
9. Error handling: HTTP 401 (auth error)
10. Error handling: Invalid input
11. Response parsing and validation
12. Configuration immutability

**Integration Tests (2):**
1. Embed-and-verify (mock OpenAI, mock network)
2. Batch efficiency (measure actual batch count)

#### Qdrant VectorStore Tests (12 unit + 2 integration)

**Unit Tests (12):**
1. Collection creation (mocked Qdrant)
2. Collection already exists (mocked response)
3. Upsert vectors with metadata (mocked response)
4. Get vector by ID (mocked response)
5. Delete vector by ID (mocked response)
6. Search by vector (mocked response)
7. Search with metadata filter (mocked response)
8. Search empty collection
9. Error: Collection not found (create on-demand)
10. Error handling: Connection timeout
11. Error handling: Auth error
12. Configuration validation

**Integration Tests (2):**
1. Full CRUD cycle (mock Qdrant client)
2. Search accuracy (verify returned results)

#### Factory Pattern Tests (6)

**Unit Tests (6):**
1. Create OpenAI provider with config
2. Create Qdrant store with config
3. List available providers
4. List available vector stores
5. Error: Unknown provider
6. Error: Unknown vector store

### Test Execution

```bash
# All tests (mocked externals, no live API calls)
pytest tests/ -v --cov=src/aegis_platform/rag --cov-report=html

# Type checking
mypy --strict src/aegis_platform/rag/

# Linting
ruff check src/aegis_platform/rag/
ruff format src/aegis_platform/rag/
```

---

## Performance Benchmarks (Target)

### Embedding Performance

| Operation | Target | Details |
|-----------|--------|---------|
| Single chunk embed | <100ms | OpenAI API call + network |
| Batch 100 chunks | <1s | OpenAI batching + network |
| Batch 1000 chunks | <10s | Multiple batched requests |

### Vector Search Performance

| Operation | Target | Details |
|-----------|--------|---------|
| Search 1K vectors | <10ms | In-memory HNSW |
| Search 1M vectors | <50ms | Indexed HNSW |
| Search 10M vectors | <100ms | Indexed HNSW + disk |
| Filtered search | <100ms | Metadata filtering doesn't degrade |

### Deployment Performance

| Resource | Target | Details |
|-----------|--------|---------|
| Docker Qdrant startup | <5s | Collection initialization |
| Connection pool warmup | <1s | Reuse connections |
| Memory overhead | <100MB | Qdrant + OpenAI client |

---

## Migration Paths

### Future: Switch Embedding Provider

```python
# Phase 3: Add Claude embeddings
EMBEDDING_PROVIDERS = {
    "openai": OpenAIEmbeddingProvider,       # Current
    "claude": ClaudeEmbeddingProvider,       # New
    "local": LocalEmbeddingProvider,         # New
}

# Configuration-driven selection (zero code changes)
EMBEDDING_PROVIDER=claude
```

### Future: Switch Vector Store

```python
# Phase 3: Add Pinecone support
VECTOR_STORES = {
    "qdrant": QdrantVectorStore,             # Current
    "pinecone": PineconeVectorStore,         # New
    "weaviate": WeaviateVectorStore,         # New
}

# Configuration-driven selection (zero code changes)
VECTOR_STORE=pinecone
```

### Future: Optimize with Caching

```
Level 1: Local cache (LRU, in-memory)
  - Cache embeddings for duplicate chunks
  - Cache search results for frequent queries

Level 2: Distributed cache (Redis)
  - Share cache across service instances
  - Reduce embedding API costs
```

---

## Quality Standards (NFR-009)

All Phase 2 implementation SHALL satisfy:

### NFR-IMPL-001: Type Safety
```
✅ All new code passes mypy --strict
✅ 100% type annotation coverage
✅ No Any types or unsafe casts
✅ Protocol compliance verified
```

### NFR-IMPL-002: Code Quality
```
✅ ruff check passes (no violations)
✅ ruff format applied (consistent style)
✅ Docstrings on all public APIs
✅ Inline comments for complex logic
```

### NFR-IMPL-003: Test Coverage
```
✅ 30+ unit + integration tests
✅ >85% code coverage
✅ All external calls mocked
✅ Edge cases covered (empty input, timeouts, errors)
```

### NFR-IMPL-004: Error Handling
```
✅ Exponential backoff with jitter
✅ Connection retry policy
✅ Immediate failure on auth errors
✅ Descriptive error messages for operators
✅ No credentials in error output
```

### NFR-IMPL-005: Configuration
```
✅ All secrets from environment variables
✅ Sensible defaults for all parameters
✅ Validation at instantiation
✅ No hardcoded credentials
✅ Clear error on missing config
```

### NFR-IMPL-006: Performance
```
✅ Embed 100 vectors in <1 second
✅ Vector search in <50ms (1M vectors)
✅ Metadata filtering doesn't degrade search
✅ Batching reduces API call count
```

---

## Implementation Checklist

### Prerequisites (Complete ✅)
- [x] RFC-009 Phase 2 sections written
- [x] SRS-001 FR-014, FR-015, FR-016, NFR-009 added
- [x] ADR-009 Phase 2 sections added
- [x] C4-001 updated with EmbedderService, VectorStoreService
- [x] EDI-001 updated with Phase 2 traceability

### Implementation (Pending)
- [ ] Create `src/aegis_platform/rag/embeddings_openai.py`
- [ ] Create `src/aegis_platform/rag/storage_qdrant.py`
- [ ] Update `src/aegis_platform/rag/provider_factory.py`
- [ ] Create configuration classes (openai_config.py, qdrant_config.py)
- [ ] Add dependencies: `uv add openai qdrant-client`
- [ ] Create comprehensive test suite (30+ tests)

### Quality Assurance (Pending)
- [ ] Type checking: `mypy --strict`
- [ ] Linting: `ruff check` + `ruff format`
- [ ] Test coverage: `pytest --cov >85%`
- [ ] Documentation: Docstrings + README + examples

### Integration (Pending)
- [ ] Integrate with existing Chunker
- [ ] End-to-end smoke tests
- [ ] Performance benchmarking
- [ ] GitHub Issues + PR review

---

## References

- **RFC-009:** AEGIS-RFC-009-RAG-Architecture.md (sections 11.2-11.8)
- **SRS-001:** AEGIS-SRS-001.md (FR-014, FR-015, FR-016, NFR-009)
- **ADR-009:** AEGIS-ADR-009-RAG-Chunking.md (Phase 2 sections)
- **C4-001:** AEGIS-C4-001.md (v1.8)
- **EDI-001:** AEGIS-EDI-001.md (v1.7)

**External References:**
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
- OpenAI Pricing: https://openai.com/pricing/#embeddings
- Qdrant Docs: https://qdrant.tech/documentation/
- Qdrant Python Client: https://github.com/qdrant/qdrant-client

---

## Experiment Scripts

The Day 10 experiment includes Python scripts to validate the implementations.

### Day 10 Experiment Script

**File:** `scripts/day10_embedding_vectorstore_experiment.py`

**Purpose:** Validate OpenAI embedding provider and Qdrant vector store implementations against performance, reliability, and quality baselines.

#### Features Tested

1. **Embedding Batching Efficiency**
   - Single chunk embedding latency
   - Batch embedding latency (10, 50, 100 chunks)
   - Latency per embedding metric
   - Batching efficiency measurement

2. **Vector Search Latency**
   - Collection creation
   - Vector insertion (100 vectors)
   - Search latency (top 10 results)
   - Metadata filtering performance

3. **Error Handling & Retry Logic**
   - Health check functionality
   - Configuration validation
   - API error handling
   - Retry mechanism verification

4. **Configuration-Driven Behavior**
   - Provider registry discovery
   - Vector store registry discovery
   - Configuration propagation
   - Multi-provider support

5. **Type Safety**
   - mypy --strict compliance
   - Type annotation coverage
   - Protocol compliance

6. **Code Quality**
   - ruff linting
   - Import ordering
   - Docstring completeness
   - Code style compliance

7. **Performance Baselines**
   - Single chunk embedding (<100ms target)
   - Batch efficiency (throughput)
   - Search latency (<50ms target on 1M vectors)

#### Usage

```bash
# Default: Use mocked API calls (no credentials needed)
uv run python scripts/day10_embedding_vectorstore_experiment.py

# Verbose output
uv run python scripts/day10_embedding_vectorstore_experiment.py --verbose

# With live OpenAI API (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... uv run python scripts/day10_embedding_vectorstore_experiment.py --live
```

#### Output

The script produces:

1. **Console Output:** Real-time test execution with pass/fail status
2. **Summary Report:** Final results for all tested components
3. **Exit Code:** 0 for success, 1 for failures

#### Sample Output

```
[============================================================]
|   Day 10 Embedding and Vector Store Experiment            |
[============================================================]

[*] Testing Embedding Batching Efficiency
============================================================
  Batch   1:   45.2ms (  45.2ms per embedding)
  Batch  10:  120.5ms (  12.1ms per embedding)
  Batch  50:  380.2ms (   7.6ms per embedding)
  Batch 100:  720.1ms (   7.2ms per embedding)

[*] Testing Vector Search Latency
============================================================
  Created collection: day10_experiment
  Inserted 100 vectors: 125.3ms
  Search (top 10): 8.7ms (found 10 results)

[OK] Type Safety (mypy --strict): PASS
[OK] Code Quality (ruff): PASS
[OK] Configuration: Available providers found
[OK] Performance: Baselines measured
```

#### Requirements

- Python 3.9+
- OpenAI Python SDK (openai)
- Qdrant Python Client (qdrant-client)
- Aegis Platform RAG module

#### Notes

- With mocked API calls (default), latencies represent internal processing only
- With live API calls (--live flag), latencies include network latency
- Qdrant must be running (Docker or cloud) for vector search tests
- No real data is persisted; collections are cleaned up after tests
- Full documentation in `scripts/README.md`

---

## Running Qdrant Locally

### Prerequisites
- Docker installed and running

### Quick Start

```bash
# Start Qdrant container (using docker-compose for convenience)
docker-compose -f docker-compose.qdrant.yml up -d

# Verify it's running
curl http://localhost:6333/health
```

**Output:**
```json
{
  "title": "qdrant - vector search engine",
  "version": "1.0.0"
}
```

### Docker Compose Configuration

The included `docker-compose.qdrant.yml` provides:
- REST API on port 6333
- gRPC API on port 6334
- Persistent storage in `qdrant_storage` volume
- Health checks
- Automatic restart

### Ports Explained

- **6333**: REST API (HTTP)
- **6334**: gRPC API (high-performance protocol)

### Verification

```bash
# Health check via REST API
curl http://localhost:6333/health

# List collections
curl http://localhost:6333/collections

# Stop Qdrant
docker-compose -f docker-compose.qdrant.yml down
```

### Python Verification

```python
import asyncio
from qdrant_client import AsyncQdrantClient

async def check_health():
    client = AsyncQdrantClient(url="http://localhost:6333")
    collections = await client.get_collections()
    print(f"Qdrant is running. Collections: {collections}")

asyncio.run(check_health())
```

### Environment Configuration

Add to `.env` or your environment:

```bash
QDRANT_URL=http://localhost:6333
QDRANT_TIMEOUT=30
QDRANT_VECTOR_SIZE=1536
QDRANT_DISTANCE_METRIC=cosine
```

### Common Issues

**Docker: "Cannot connect to Docker daemon"**
- Start Docker Desktop (macOS/Windows) or run `sudo systemctl start docker` (Linux)

**Port Already in Use**
```bash
# Windows: Kill process on port 6333
netstat -ano | findstr :6333
taskkill /PID <PID> /F

# Linux/macOS:
lsof -i :6333
kill -9 <PID>
```

**Qdrant Connection Refused**
```bash
# Check if running
curl http://localhost:6333/health

# If not, start it
docker-compose -f docker-compose.qdrant.yml up -d
```

### Full RAG Pipeline Example

```python
from aegis_platform.rag import (
    ChunkingConfig, create_chunker,
    OpenAIEmbeddingConfig, create_embedding_provider,
    QdrantVectorStoreConfig, create_vector_store,
)

async def rag_pipeline():
    # 1. Chunk documents (Phase 1)
    chunker = create_chunker(
        ChunkingConfig(strategy="recursive", chunk_size=512)
    )
    chunks = chunker.chunk("Your long document text...")

    # 2. Create embeddings (Phase 2)
    embedder = create_embedding_provider(
        OpenAIEmbeddingConfig(api_key="sk-...")
    )
    embeddings = await embedder.embed([c.text for c in chunks])

    # 3. Store in Qdrant (Phase 2)
    store = create_vector_store(
        QdrantVectorStoreConfig(url="http://localhost:6333")
    )
    await store.insert(
        collection="documents",
        vectors=embeddings,
        metadata=[{"id": c.id, "source": "docs"} for c in chunks],
    )

    # 4. Query (Phase 3 - future)
    query = "What is RAG?"
    query_embedding = await embedder.embed([query])
    results = await store.search(
        collection="documents",
        vector=query_embedding[0],
        limit=5,
    )
    return results
```

---
