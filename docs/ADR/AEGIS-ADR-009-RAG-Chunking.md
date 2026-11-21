# AEGIS-ADR-009

# Architecture Decision Record - RAG Foundation Chunking Architecture

**Status:** Accepted

**Date:** 2026-11-15

**Owner:** Kader Beevi

**Related Documents:**
- SRS: AEGIS-SRS-001 (RAG Foundation section)
- RFC: AEGIS-RFC-009 (RAG Architecture)
- EDI: AEGIS-EDI-001

---

## Context

The Aegis platform requires a Retrieval-Augmented Generation (RAG) capability to augment LLM reasoning with retrieved documents. The foundation phase must establish:

1. Flexible, production-ready text chunking for various document types
2. Abstract interfaces for embeddings and vector storage (provider-agnostic)
3. Configuration-driven dependency injection for strategy selection
4. Extensible architecture for future chunking strategies and vector store implementations

**Challenge:** Text chunking is domain-specific (code vs. markdown vs. general text requires different approaches), and vector storage providers vary significantly (Qdrant, Weaviate, Pinecone, Chroma, etc.). A monolithic implementation would lock us into specific choices.

**Constraint:** Must be production-ready with no stub implementations or future TODOs in critical paths.

---

## Options Considered

### Option A: Custom-Only Chunking

**Approach:** Implement 3 lightweight custom chunking strategies (FixedSize, Recursive, Token-based).

**Pros:**
- Full control over implementation
- No external dependencies
- Lightweight and fast

**Cons:**
- Custom implementations less battle-tested than open-source alternatives
- Maintenance burden on edge cases (Unicode, special characters)
- Duplicates work already done in LangChain
- Limited strategy variety (3 strategies insufficient for all use cases)

**Decision:** Rejected - insufficient strategy variety and maintenance risk

---

### Option B: LangChain-Only Chunking

**Approach:** Use only LangChain's text splitters (RecursiveCharacterTextSplitter, MarkdownTextSplitter, etc.).

**Pros:**
- Battle-tested implementations
- Lower maintenance burden
- Rich strategy ecosystem
- Direct integration with LangChain ecosystem

**Cons:**
- Adds external dependency (though LangChain already transitive via langgraph)
- Less control over implementation
- Possible over-engineering for simple cases

**Decision:** Accepted - Production readiness outweighs concerns. LangChain is battle-tested in production systems.

---

### Option C: Hybrid (Custom + LangChain)

**Approach:** Keep 3 custom strategies + add 2 LangChain wrappers. Users choose strategy via config.

**Pros:**
- Best of both: custom for simple cases, LangChain for complex
- Maximum flexibility
- Enables A/B testing strategies

**Cons:**
- Maintenance burden increases
- Test coverage becomes complex
- Potential confusion on which strategy to use when

**Decision:** Initially explored but rejected after user requested simplification. Hybrid adds complexity without proportional benefit.

---

## Decision

**Implement RAG Foundation using LangChain-only chunking strategies with provider-agnostic architecture.**

### Rationale

1. **Production Readiness**: LangChain's text splitters are battle-tested in production systems. No need to reinvent.

2. **Extensibility**: Abstract protocol-based architecture allows adding new strategies without modification to existing code or tests.

3. **Vendor Neutrality**: Abstract `EmbeddingProvider` and `VectorStore` protocols ensure we don't lock into specific providers. Implementations can be swapped at runtime.

4. **Configuration-Driven**: `ChunkingConfig` enables strategy selection via environment variables without code changes.

5. **Simplicity**: Two well-designed strategies (Recursive, Markdown) cover 90% of use cases. Future strategies can be added incrementally.

### Implementation Details

#### Chunking Architecture

```
ChunkingStrategy (Abstract Protocol)
├── LangChainRecursiveChunker
│   └── Wraps: LangChain RecursiveCharacterTextSplitter
├── LangChainMarkdownChunker
│   └── Wraps: LangChain MarkdownTextSplitter
└── Future: SemanticChunker, SlidingWindowChunker, etc.

ChunkingConfig (Configuration Object)
└── strategy: str (name) + chunk_size, overlap, min/max sizes

create_chunker() (Factory)
└── Instantiates correct strategy based on config
```

#### Protocol-Based Storage

```
EmbeddingProvider (Abstract Protocol)
├── No concrete implementations (Day 2)
└── Future: OpenAIEmbedder, ClaudeEmbedder, LocalEmbedder

VectorStore (Abstract Protocol)
├── No concrete implementations (Day 2)
└── Future: QdrantStore, WeaviateStore, PineconeStore
```

#### Factory Pattern

```python
# Register strategies
CHUNKING_STRATEGIES = {
    "recursive": LangChainRecursiveChunker,
    "markdown": LangChainMarkdownChunker,
}

# Create based on config
def create_chunker(config: ChunkingConfig) -> ChunkingStrategy:
    strategy_class = CHUNKING_STRATEGIES[config.strategy]
    return strategy_class(chunk_size=config.chunk_size, ...)
```

---

## Trade-offs

### Advantages

1. **Production Quality**: Using battle-tested LangChain implementations reduces bugs and edge cases.

2. **Maintainability**: One less codebase to maintain. Security updates handled by LangChain.

3. **Extensibility**: Protocol-based design allows infinite strategies without core changes.

4. **Runtime Flexibility**: Configuration enables A/B testing and adaptation without code changes.

5. **Type Safety**: Full type annotations with mypy validation. No Any or unsafe casts.

6. **Testing**: 24 comprehensive tests covering initialization, validation, chunking, configuration, metadata, edge cases, and integration scenarios.

7. **Documentation**: Clear README with usage examples, API docs, and future roadmap.

### Disadvantages

1. **Dependency**: Adds `langchain-text-splitters` dependency (though LangChain is already transitive).

2. **Less Control**: Can't optimize LangChain implementation for specific edge cases.

3. **Potential Over-engineering**: For very simple fixed-size chunking, LangChain may be overkill.

4. **Learning Curve**: Developers must understand LangChain's text splitter API.

---

## Consequences

### Immediate (Implemented)

1. ✅ Two production-ready chunking strategies available
2. ✅ Abstract protocols for embeddings and vector storage
3. ✅ Configuration-driven dependency injection working
4. ✅ 24 unit tests, all passing
5. ✅ Type checking and linting clean
6. ✅ Documentation complete

### Short Term (Day 3)

1. Implement concrete `EmbeddingProvider` classes (OpenAI, Claude)
2. Implement concrete `VectorStore` classes (Qdrant)
3. Build integration tests between chunking and embeddings
4. Create end-to-end RAG pipeline smoke tests

### Medium Term (Day 4+)

1. Add document loader for various file formats
2. Implement batch indexing pipeline
3. Explore advanced strategies (semantic chunking, sliding window)
4. Add performance optimization and caching

### Long Term (Production)

1. RAG becomes core agent capability for context-aware reasoning
2. Support for multiple vector stores and embedding providers
3. Retrieval ranking and reranking strategies
4. Evaluation metrics for chunk quality and retrieval accuracy

---

## Risk Mitigation

### Dependency Risk
- **Mitigation**: LangChain-text-splitters is small, stable package. Used in production at scale.

### Change Risk
- **Mitigation**: Protocol-based architecture allows swapping implementations if LangChain breaks.

### Complexity Risk
- **Mitigation**: Clear separation of concerns. Each component (chunking, embeddings, storage) is independent.

### Future Extensibility Risk
- **Mitigation**: Protocol-based design proven extensible. Can add strategies without core changes.

---

## Validation

### Testing

- 24 unit tests covering all acceptance criteria
- All tests passing
- Type checking clean (mypy)
- Linting clean (ruff)

### Smoke Tests

```python
# Recursive strategy works
config = ChunkingConfig(strategy="recursive", chunk_size=512)
chunker = create_chunker(config)
chunks = chunker.chunk("Text...", "doc-1")
assert len(chunks) > 0

# Markdown strategy works
config = ChunkingConfig(strategy="markdown", chunk_size=1024)
chunker = create_chunker(config)
chunks = chunker.chunk("# Title\n\nContent...", "doc-2")
assert len(chunks) > 0

# Factory errors on invalid strategy
with pytest.raises(ValueError):
    create_chunker(ChunkingConfig(strategy="invalid"))
```

### Documentation

- README with architecture, usage examples, testing instructions
- SRS with functional and non-functional requirements
- Inline code documentation with docstrings
- Clear error messages for configuration errors

---

## Alternative Futures

### If Requirement Changes

1. **Need custom strategy**: Implement `ChunkingStrategy` protocol, register in factory
2. **Need different vector store**: Implement `VectorStore` protocol, update config
3. **Need embedding provider**: Implement `EmbeddingProvider` protocol, update factory

None of these require core architecture changes.

---

## Alignment with Aegis Principles

✅ **Documentation First:** SRS-001 v1.7 (FR-013, NFR-008) defines all requirements before implementation

✅ **Extensible Design:** Protocol-based architecture proven extensible

✅ **Production Ready:** Battle-tested LangChain implementations, comprehensive testing

✅ **Type Safe:** Full type annotations, mypy clean

✅ **Testable:** 24 tests covering happy path, edge cases, integration

✅ **Maintainable:** Clear separation of concerns, single responsibility

---

## Related Decisions

- ADR-001: Adopted FastAPI for API foundation (similar philosophy: proven, production-ready)
- ADR-003: Adopted LangGraph for agent orchestration (similar: battle-tested framework)
- ADR-004: LLM provider abstraction (similar: protocol-based provider agnosticism)

This ADR extends the pattern: use battle-tested frameworks + abstract interfaces = production-ready + extensible.

---

## Approval Sign-Off

- **Decision Made By:** Architecture Review
- **Approval Date:** 2026-11-15
- **Implementation Status:** ✅ COMPLETE
- **Next Review:** Day 3 (Embedding Provider + Vector Store implementations)

---

---

# Phase 2: Embedding Provider & Vector Store Selection

## Context for Phase 2

With chunking foundation complete (Phase 1), Phase 2 implements concrete embedding and vector storage providers to complete the end-to-end RAG pipeline.

**Decisions Required:**
1. Which embedding provider? (OpenAI, Claude, local models, Cohere, etc.)
2. Which vector store? (Qdrant, Pinecone, Weaviate, Milvus, etc.)

Both decisions must maintain the provider-agnostic architecture established in Phase 1.

---

## Phase 2.1: Embedding Provider Decision

### Options Considered

#### Option A: OpenAI text-embedding-3-small (Selected)

**Model:** `text-embedding-3-small`
- **Dimensions:** 1536
- **Cost:** ~$0.02 per 1M tokens (lowest among OpenAI models)
- **Speed:** Sub-100ms typical latency
- **Features:** Native batching support (up to 2048 embeddings per request)
- **Stability:** Production-ready (released 2024)

**Pros:**
- Most cost-effective OpenAI model
- Native batching reduces API calls and latency
- Proven track record in production systems
- Aligns with enterprise adoption
- Easy to swap to text-embedding-3-large later if needed
- Handles 8,191 token context window (sufficient for chunked text)

**Cons:**
- Requires OpenAI API key and account
- Rate limits apply (lower-tier accounts ~10 req/sec)
- Adds dependency on external vendor
- Slightly lower quality than text-embedding-3-large (acceptable for RAG)

#### Option B: Local Embeddings (sentence-transformers)

**Model:** `all-MiniLM-L6-v2` or similar
- **Dimensions:** 384–768
- **Cost:** Zero (open-source)
- **Speed:** ~50ms per batch on CPU
- **Features:** GPU acceleration optional

**Pros:**
- No external API calls needed
- No rate limiting concerns
- No vendor lock-in
- Privacy (embeddings stay on-premise)

**Cons:**
- Requires model download and disk space
- Slower than cloud alternatives (without GPU)
- Requires maintenance and updates
- Quality varies by model choice
- Adds memory overhead to application
- Not enterprise-grade for compliance

#### Option C: Alternative Cloud Providers (Cohere, Voyage AI)

**Cohere embed-english-v3.0**
- Similar cost to OpenAI
- Good quality and batching

**Voyage AI voyage-large-2**
- Higher quality, higher cost (~$0.12 per 1M tokens)
- Less adoption in Aegis context

**Pros:**
- Viable alternatives with comparable features
- Support for future diversification

**Cons:**
- Higher cost than OpenAI
- Smaller ecosystem and community support
- Adds unnecessary complexity for Phase 2
- Delays time-to-production

---

### Decision: OpenAI text-embedding-3-small

**Rationale:**
1. **Cost Leadership:** Lowest-cost OpenAI option ($0.02 per 1M tokens)
2. **Batch Efficiency:** Native support for up to 2,048 embeddings per request
3. **Production Reliability:** Widely used in production RAG systems
4. **Protocol Compliance:** Aligns with RFC-009 EmbeddingProvider protocol
5. **Migration Path:** Easy to swap to text-embedding-3-large or another provider

### Implementation (Phase 2)

The EmbeddingProvider implementation shall:
1. Use `openai.OpenAI` Python client library
2. Support batching (up to 2048 embeddings per request)
3. Implement exponential backoff retry for rate limiting (HTTP 429)
4. Validate API key presence at instantiation
5. Return standardized `Embedding` protocol objects
6. Be fully testable via mocked OpenAI responses

**File:** `src/aegis_platform/rag/embeddings_openai.py`
**Tests:** 12 unit + 2 integration tests
**Dependencies:** `uv add openai`

---

### Trade-offs: Embedding Provider

**Advantages:**
- Cost effective; sub-100ms latency
- Native batching reduces overhead
- Protocol compliance (no vendor-specific leakage)
- Testable without real API calls

**Disadvantages:**
- Vendor lock-in risk
- Rate limits; network dependency
- API key management required
- Cost opacity

---

## Phase 2.2: Vector Store Decision

### Options Considered

#### Option A: Qdrant (Selected)

**Technology:** Rust-based, open-source vector database
- **Deployment:** Docker, cloud-hosted, on-premise
- **Protocol:** gRPC + REST API
- **Scaling:** Horizontal scaling with clustering
- **Search Quality:** Approximate Nearest Neighbor (ANN) with HNSW algorithm
- **Maturity:** Production-ready (v1.0+ released 2023)

**Pros:**
- **Performance:** Sub-10ms search latency on millions of vectors
- **Flexibility:** HNSW indexing + metadata filtering
- **Open Source:** Fully transparent, no vendor lock-in
- **Deployment Options:** Cloud (Qdrant Cloud), self-hosted Docker, Kubernetes
- **Python SDK:** Well-maintained `qdrant-client` library
- **Payload/Metadata:** First-class support for rich metadata
- **Cost:** No per-query charges; pay only for infrastructure
- **Community:** Strong adoption in open-source RAG (LangChain, LlamaIndex)

**Cons:**
- **Operational Complexity:** Requires database management for on-premise
- **Clustering Setup:** Enterprise clustering needs manual configuration
- **Learning Curve:** Distinct API compared to SaaS platforms
- **Warm-up Time:** HNSW index construction requires upfront time

#### Option B: Pinecone (SaaS)

**Technology:** Fully managed vector database
- **Deployment:** Cloud-only (no on-premise)
- **Scaling:** Automatic, transparent
- **Search Quality:** Proprietary indexing, very fast
- **Maturity:** Production-ready, widely adopted

**Pros:**
- Simplicity; fully managed
- Automatic scaling
- Zero operations
- Fast

**Cons:**
- Vendor lock-in (proprietary API)
- Cost structure (per-query charges)
- No on-premise option
- System depends on Pinecone availability
- Data privacy concerns (GDPR/HIPAA)
- Frequent API changes (breaking compatibility)

#### Option C: Weaviate

**Technology:** Golang-based, open-source vector database
- **Deployment:** Docker, cloud-hosted, on-premise
- **Protocol:** REST API + GraphQL
- **Scaling:** Clustering via Raft
- **Search Quality:** HNSW + proprietary indexing

**Pros:**
- GraphQL support
- Cloud option available
- Flexible schema
- Growing community

**Cons:**
- Higher memory overhead than Qdrant
- Smaller ecosystem
- Less comprehensive documentation
- GraphQL adds complexity

#### Option D: Milvus

**Technology:** Golang-based, open-source vector database
- **Deployment:** Docker, cloud-hosted (Zilliz Cloud), on-premise
- **Scaling:** Horizontal scaling built-in
- **Search Quality:** Multiple indexing algorithms

**Pros:**
- Native horizontal scaling
- Cost-effective at scale
- Fast search
- Flexible index types

**Cons:**
- Complex operational setup
- Less clear documentation
- Smaller Python community
- Steeper learning curve

#### Option E: Chroma (Embedded)

**Technology:** Python-based, lightweight storage
- **Deployment:** In-process (no separate database)
- **Scaling:** Single machine only
- **Search Quality:** Brute-force indexing

**Pros:**
- Easy setup
- No operations
- Python native

**Cons:**
- Not production-grade
- ~100K vectors maximum
- No clustering
- Slow search
- No replication
- Demos only, not suitable for Phase 2

---

### Decision: Qdrant

**Rationale:**
1. **Performance:** Sub-10ms search on millions of vectors
2. **Flexibility:** On-premise + cloud deployment
3. **Open Source:** Transparent; no vendor lock-in
4. **Metadata:** First-class support (critical for RAG)
5. **Ecosystem:** Wide adoption in LangChain/LlamaIndex

### Implementation (Phase 2)

The VectorStore implementation shall:
1. Use `qdrant-client` Python library
2. Support collection management (create, delete, exists)
3. Support CRUD: upsert vectors with metadata, get by ID, delete by ID
4. Support vector similarity search with metadata filtering
5. Store payload (metadata) alongside vectors
6. Return standardized `SearchResult` protocol objects
7. Be deployable via Docker or Qdrant Cloud
8. Be fully testable via in-memory Qdrant instance

**File:** `src/aegis_platform/rag/storage_qdrant.py`
**Tests:** 12 unit + 2 integration tests
**Dependencies:** `uv add qdrant-client`

---

### Trade-offs: Vector Store

**Advantages:**
- Open source & transparent
- Sub-10ms search performance
- On-premise & cloud flexible
- Metadata-first architecture
- Cost predictability
- Wide ecosystem adoption

**Disadvantages:**
- Operational complexity
- Scaling requires understanding clustering
- Memory overhead for index construction
- Learning curve vs. managed SaaS

---

## Phase 2.3: Factory Pattern Extension

### Factory Registry

The factory from Phase 1 extends to support embedding and vector store providers:

```python
# Embedding provider registry
EMBEDDING_PROVIDERS = {
    "openai": OpenAIEmbeddingProvider,
    # Future: "claude": ClaudeEmbeddingProvider,
    # Future: "local": LocalEmbeddingProvider,
}

# Vector store registry
VECTOR_STORES = {
    "qdrant": QdrantVectorStore,
    # Future: "pinecone": PineconeVectorStore,
    # Future: "weaviate": WeaviateVectorStore,
}


# Factory functions
def create_embedding_provider(name: str, config: EmbeddingConfig) -> EmbeddingProvider: ...


def create_vector_store(name: str, config: VectorStoreConfig) -> VectorStore: ...
```

**File:** `src/aegis_platform/rag/provider_factory.py` (extend existing)
**Tests:** 6 unit tests for factory instantiation

---

## Phase 2: Quality Standards (NFR-009)

All Phase 2 implementation must satisfy:

### NFR-IMPL-001: Type Safety
- All new code passes `mypy --strict`
- All parameters and return types fully annotated

### NFR-IMPL-002: Code Quality
- All new code passes `ruff` linting
- Docstrings present for all public APIs

### NFR-IMPL-003: Test Coverage
- 30+ unit and integration tests
- Coverage >85% for all new code
- All external API calls mocked in tests

### NFR-IMPL-004: Error Handling & Retry
- HTTP 429 (rate limit): Exponential backoff with jitter
- Connection errors: Retry per configuration
- HTTP 401 (auth): Fail immediately
- Descriptive error messages for operators

### NFR-IMPL-005: Configuration & Secrets
- All secrets from environment variables
- Configuration defaults sensible
- Validation at instantiation
- No hardcoded credentials

### NFR-IMPL-006: Performance
- Embedding batch <1 second for 100 vectors
- Vector search <50ms in 1M vectors (indexed)
- Metadata filtering doesn't degrade search

---

## Phase 2: Implementation Status

| Component | Status | Details |
|-----------|--------|---------|
| OpenAI EmbeddingProvider | 🔄 In Progress | embeddings_openai.py |
| Qdrant VectorStore | 🔄 In Progress | storage_qdrant.py |
| Factory Extension | 🔄 In Progress | provider_factory.py (skeleton exists) |
| Configuration | 🔄 In Progress | Config objects with validation |
| Tests (30+) | 🔄 Pending | OpenAI (12), Qdrant (12), Factory (6) |
| Type Checking | 🔄 Pending | mypy --strict |
| Linting | 🔄 Pending | ruff clean |
| Documentation | 🔄 Pending | Docstrings, README, examples |

---

## Phase 2: Consequences

### Immediate (Phase 2)

1. **Dependencies Added:** `uv add openai qdrant-client`
2. **Configuration Required:**
   - `OPENAI_API_KEY` (required)
   - `OPENAI_EMBEDDING_MODEL` (default: text-embedding-3-small)
   - `OPENAI_BATCH_SIZE` (default: 1024)
   - `QDRANT_URL` (default: http://localhost:6333)
   - `QDRANT_COLLECTION_NAME` (default: aegis_embeddings)
3. **Deployment:** Docker Qdrant or Qdrant Cloud

### Short Term (Phase 3+)

1. Provider diversification (Claude, local models, Pinecone, Weaviate)
2. Cost monitoring via observability layer
3. Embedding caching for duplicate chunks
4. Advanced features (hybrid search, reranking)

### Long Term

1. Migration paths if requirements change
2. Compliance & privacy requirements (on-premise Qdrant for GDPR)
3. Performance optimization at scale

---

## Phase 2: Alignment with Aegis Principles

✅ **Protocol-Based Abstraction:** EmbeddingProvider and VectorStore remain provider-agnostic; only implementations change

✅ **Configuration-Driven:** Environment variables enable provider selection without code changes

✅ **Production Ready:** OpenAI and Qdrant are battle-tested in production systems

✅ **Type Safe:** Full type annotations, mypy --strict

✅ **Testable:** All external calls mocked in CI/CD; no live API calls in tests

✅ **Extensible:** New embedding providers and vector stores can be added by implementing protocols and registering in factory

---

## Phase 2: Decision Cross-Reference

| Stage | Reference |
|--------|-----------|
| RFC | AEGIS-RFC-009 (sections 11.2-11.8) |
| SRS | AEGIS-SRS-001 (FR-014, FR-015, FR-016, NFR-009) |
| C4 | AEGIS-C4-001 (EmbedderService, VectorStoreService containers) |
| EDI | AEGIS-EDI-001 (Phase 2 traceability) |
| Implementation | TBD (Pending approval) |

---

## References

- **OpenAI Embeddings:** https://platform.openai.com/docs/guides/embeddings
- **OpenAI Pricing:** https://openai.com/pricing/#embeddings (as of 2026-11-15)
- **Qdrant Documentation:** https://qdrant.tech/documentation/
- **Qdrant Python Client:** https://github.com/qdrant/qdrant-client
- **Qdrant Pricing:** https://qdrant.tech/documentation/pricing/
- **RFC-009:** AEGIS-RFC-009-RAG-Architecture.md
- **SRS-001:** AEGIS-SRS-001.md

---

## Approval Sign-Off

- **Phase 1 Status:** ✅ COMPLETE (Chunking + Protocols + Factory)
- **Phase 2 Status:** 🔄 IN PROGRESS (Embedding + Vector Store decisions made; implementation pending)
- **Decisions Made By:** Architecture Review
- **Approval Date:** 2026-11-15
- **Next Review:** Implementation validation (Phase 2 code, tests, quality)
