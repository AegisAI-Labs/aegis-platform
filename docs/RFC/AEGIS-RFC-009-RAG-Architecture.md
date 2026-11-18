# RFC-009: RAG Architecture

- **Status:** Accepted
- **Owner:** AEGIS Platform
- **Date:** 2026-11-15
- **Scope:** Retrieval-Augmented Generation (RAG) foundation and provider-agnostic abstractions

## 1. Problem Statement

Large Language Models have fixed knowledge cutoffs and cannot reason about proprietary or real-time data without augmentation.

Retrieval-Augmented Generation (RAG) addresses this by:

1. Storing documents in a vector database
2. Retrieving relevant documents at query time
3. Injecting retrieved context into the LLM prompt
4. Allowing the LLM to reason over current, domain-specific information

However, RAG implementation involves multiple vendor-specific choices:

- **Text chunking:** Different document types (code, markdown, prose) require different chunking strategies
- **Embeddings:** Multiple embedding providers (OpenAI, Claude, local models)
- **Vector storage:** Multiple vector databases (Qdrant, Weaviate, Pinecone, Chroma)

A tightly-coupled implementation locks the platform into specific vendors.

The AEGIS platform requires:

- Provider-agnostic abstractions for embeddings and vector storage
- Extensible text chunking strategies
- Configuration-driven provider selection
- Production-ready implementations (no stubs or TODOs)

## 2. Goals

1. Define abstract protocols for embedding providers and vector stores
2. Implement production-ready text chunking with multiple strategies
3. Support provider selection at runtime via configuration
4. Enable future provider implementations without core changes
5. Provide comprehensive testing and documentation
6. Maintain type safety and observability throughout the stack

## 3. Non-Goals

- Document loading and ingestion (future work)
- RAG evaluation and retrieval quality metrics (future work)
- Integration with agent service (future work)
- Multi-language or non-text document support (future work)

**Phase 2 Note:** Concrete embedding provider (OpenAI) and vector store (Qdrant) implementations are now IN SCOPE (Day 10), updated from "future work".

## 4. Proposed Architecture

### 4.1 Three-Layer Design

```
Application Layer
    |
    v
RAG Service (query, insert, search)
    |
    +---- EmbeddingProvider (protocol) ---- [Phase 2: OpenAI, Phase 3+: Claude, Local]
    |
    +---- ChunkingStrategy (protocol) ---- [Phase 1: LangChain Recursive, Markdown]
    |
    +---- VectorStore (protocol) -------- [Phase 2: Qdrant, Phase 3+: Weaviate, Pinecone]
    |
    v
Configuration
    (Provider selection, credentials, endpoints, chunk settings)
```

### 4.2 Core Components

#### ChunkingStrategy (Abstract Protocol)
- **Purpose:** Split documents into overlapping chunks for embedding and retrieval
- **Implementations:**
  - LangChainRecursiveChunker (recursive structure-aware chunking)
  - LangChainMarkdownChunker (markdown-aware chunking)
- **Future implementations:** Code-aware, HTML, PDF, multi-language
- **Key methods:**
  - `chunk(text: str, doc_id: str) -> List[Chunk]`
  - `get_config() -> ChunkingConfig`
  - `update_config(**kwargs) -> None`

#### EmbeddingProvider (Abstract Protocol)
- **Purpose:** Convert text to vector embeddings
- **Design pattern:** Provider-neutral contract
- **Future implementations:**
  - OpenAI embeddings (text-embedding-3-small, text-embedding-3-large)
  - Claude embeddings
  - Local embeddings (sentence-transformers, ONNX)
- **Key methods:**
  - `embed(text: str) -> np.ndarray`
  - `embed_batch(texts: List[str]) -> List[np.ndarray]`
  - `get_config() -> EmbeddingConfig`

#### VectorStore (Abstract Protocol)
- **Purpose:** Store, search, and manage vector embeddings
- **Design pattern:** Provider-neutral contract
- **Future implementations:**
  - Qdrant (native vector database)
  - Weaviate (vector database with filtering)
  - Pinecone (managed vector database)
  - Chroma (lightweight local option)
- **Key methods:**
  - `insert(vectors: List[Vector]) -> None`
  - `search(query_vector: np.ndarray, top_k: int) -> List[SearchResult]`
  - `delete(vector_ids: List[str]) -> None`
  - `get_config() -> VectorStoreConfig`

#### Chunk Model
```python
@dataclass
class Chunk:
    text: str                           # Content of the chunk
    chunk_id: str                       # Format: doc_id#sequence
    sequence: int                       # Position in document
    start_pos: int                      # Character position in original
    end_pos: int                        # Character position in original
    metadata: Dict[str, Any]            # Strategy, document_id, source, etc.
```

#### Document Model
```python
@dataclass
class Document:
    doc_id: str                         # Unique document identifier
    content: str                        # Full document text
    metadata: Dict[str, Any]            # Source, type, timestamp, etc.
```

### 4.3 Factory Pattern

```python
def create_chunker(config: ChunkingConfig) -> ChunkingStrategy:
    """
    Instantiate a chunking strategy based on configuration.

    Args:
        config: ChunkingConfig with strategy name and parameters

    Returns:
        Configured chunker instance

    Raises:
        ValueError: If strategy not found or config invalid
    """
```

**Strategy Registry:**
```python
CHUNKING_STRATEGIES = {
    "recursive": LangChainRecursiveChunker,
    "markdown": LangChainMarkdownChunker,
    # Future: "code", "html", "pdf", etc.
}
```

## 5. Configuration

### ChunkingConfig (Immutable)
```python
@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "recursive"         # Strategy name
    chunk_size: int = 1024              # Target chunk size in characters
    overlap: int = 128                  # Overlap between chunks
    min_chunk_size: int = 50            # Minimum chunk size
    max_chunk_size: int = 10000         # Maximum chunk size
```

### EmbeddingConfig (Future)
```python
@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "openai"            # Provider name
    model: str = "text-embedding-3-small"
    api_key: str = ""                   # From environment
    base_url: str = ""                  # Optional custom endpoint
    timeout: int = 30
```

### VectorStoreConfig (Future)
```python
@dataclass(frozen=True)
class VectorStoreConfig:
    type: str = "qdrant"                # Vector store type
    url: str = "http://localhost:6333"  # Connection URL
    collection_name: str = "documents"  # Collection/index name
    distance_metric: str = "cosine"     # Distance metric
    api_key: str = ""                   # Optional auth
```

## 6. Data Flow

### Document Ingestion (Future)
```
Document
    |
    v
ChunkingStrategy.chunk()
    |
    v
List[Chunk] (with positions, metadata)
    |
    v
EmbeddingProvider.embed_batch()
    |
    v
List[Vector] (chunk text -> embedding)
    |
    v
VectorStore.insert()
    |
    v
Persisted in vector database
```

### Query Execution (Future)
```
User Query
    |
    v
EmbeddingProvider.embed()
    |
    v
Query Vector
    |
    v
VectorStore.search(top_k=5)
    |
    v
List[SearchResult] (most similar chunks)
    |
    v
Inject into LLM Prompt
    |
    v
LLMProvider.generate()
    |
    v
Augmented Response
```

## 7. Design Decisions

### Decision 1: Protocol-Based Abstractions
**Rationale:** `@runtime_checkable` protocols allow:
- Runtime duck-typing without inheritance
- Easy mocking for tests
- No coupling to vendor SDKs in core module
- Future providers can be added without modifying existing code

**Alternative considered:** Abstract base classes (ABC)
- Would require subclassing, tighter coupling
- Less flexible for external providers

### Decision 2: LangChain-Based Chunking (Phase 1)
**Rationale:**
- Battle-tested implementations with edge case handling
- Rich strategy ecosystem (recursive, markdown, code-aware, etc.)
- Reduced maintenance burden vs. custom implementations
- Lower risk of Unicode/special character bugs

**Alternative considered:** Custom-only chunking
- Would reduce external dependencies
- Custom implementations are error-prone and less maintainable

### Decision 3: Immutable Configuration After Instantiation
**Rationale:**
- Prevents inconsistent state during operations
- Encourages explicit reconfiguration via re-instantiation
- Simpler reasoning about chunker behavior
- Works well with functional pipelines

**Alternative considered:** Mutable configuration
- Would allow runtime parameter changes
- Risk of race conditions, inconsistent state

### Decision 4: Factory Pattern for Strategy Selection
**Rationale:**
- Single point of strategy instantiation
- Centralized validation and error handling
- Easy to extend with new strategies
- Supports configuration-driven behavior

## 8. Testing Strategy

### Unit Tests (24 tests, all passing)
- ChunkingConfig validation (chunk_size, overlap, limits)
- Recursive chunker (basic, paragraphs, edge cases)
- Markdown chunker (code blocks, headers, formatting)
- Factory function (creation, invalid strategy, registry)
- Integration (all strategies handle same text consistently)

### Test Coverage
- Happy path: Basic chunking scenarios
- Edge cases: Empty text, very small chunks, overlaps > size
- Error handling: Invalid configuration, missing strategies
- Metadata tracking: Chunk IDs, positions, strategy info
- Reconfiguration: update_config() creates new splitter

### Future Test Coverage
- Embedding provider contracts (mock implementations)
- Vector store CRUD operations
- End-to-end RAG pipeline (chunk -> embed -> search)
- Performance benchmarks (chunking speed, embedding latency)

## 9. Observability

### Metrics (Future)
```
rag_chunks_created_total          # Total chunks produced
rag_chunk_size_bytes              # Size distribution
rag_embedding_requests_total      # Embedding requests
rag_embedding_latency_seconds     # Embedding latency
rag_search_requests_total         # Vector search requests
rag_search_latency_seconds        # Search latency
```

### Structured Events (Future)
```
rag.document.chunked              # Document chunked, chunk count
rag.embedding.generated           # Embeddings created for chunk
rag.search.completed              # Search query executed
rag.search.cache_hit              # Search cache hit (if applicable)
```

### Privacy & Security
- No prompts, completions, or raw document text in logs
- Chunk IDs, counts, and latencies are safe to log
- Document content only in error debug logs
- API keys stored in environment, never logged

## 10. Deployment & Configuration

### Environment Variables
```bash
# Chunking strategy
RAG_CHUNKING_STRATEGY=recursive          # "recursive" or "markdown"
RAG_CHUNK_SIZE=1024                      # Target chunk size
RAG_CHUNK_OVERLAP=128                    # Overlap between chunks
RAG_MIN_CHUNK_SIZE=50
RAG_MAX_CHUNK_SIZE=10000

# Embedding provider (future)
RAG_EMBEDDING_PROVIDER=openai
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_EMBEDDING_API_KEY=sk-...             # From environment

# Vector store (future)
RAG_VECTOR_STORE_TYPE=qdrant
RAG_VECTOR_STORE_URL=http://localhost:6333
RAG_VECTOR_STORE_COLLECTION=documents
```

### Runtime Selection
```python
from aegis_platform.rag.chunking import create_chunker
from aegis_platform.rag.models import ChunkingConfig

config = ChunkingConfig(
    strategy="recursive",           # From env
    chunk_size=1024,               # From env
    overlap=128,                   # From env
)
chunker = create_chunker(config)
chunks = chunker.chunk(document_text, doc_id="doc_123")
```

## 11. Phase 2 Implementation: EmbeddingProvider & VectorStore Concrete Implementations

### 11.1 OpenAI EmbeddingProvider

**Architectural Role:**
- Implements EmbeddingProvider protocol from Phase 1
- Translates text sequences to dense vector representations
- Supports single and batch operations for efficiency
- Handles provider-specific concerns (authentication, rate limiting, retries)

**Key Design Attributes:**
- Configurable model selection and dimensionality via environment variables
- Graceful handling of rate limits with exponential backoff
- Input validation and graceful handling of edge cases
- Observable health status for operational monitoring
- Clear error messages for authentication and connectivity issues

**Requirements:**
- Support both synchronous and asynchronous API patterns
- Provide batch operation capability with configurable batch size
- Return vector dimension for downstream vector store configuration
- Implement protocol contracts fully (no stub implementations)

### 11.2 Qdrant VectorStore

**Architectural Role:**
- Implements VectorStore protocol from Phase 1
- Manages distributed storage and retrieval of vector embeddings
- Supports collection-based organization of vectors
- Provides similarity search capabilities for RAG retrieval
- Handles metadata-enriched storage for chunk context

**Key Design Attributes:**
- REST API-based connectivity for distributed deployment
- Configurable distance metrics (cosine, euclidean, dot product)
- Collection management (create, list, stats) for multi-document scenarios
- Metadata support for preserving chunk context and filtering
- Dimension validation to ensure embedding compatibility
- Observable health status for operational monitoring

**Requirements:**
- Support CRUD operations (insert, search, delete, update) on vector collections
- Provide batch insertion capability for performance
- Enable similarity search with configurable result limits and thresholds
- Support optional metadata filtering before similarity search
- Implement protocol contracts fully (no stub implementations)

### 11.3 Factory Pattern Extension

**Architectural Role:**
- Provides centralized instantiation of embedding providers and vector stores
- Abstracts provider selection logic for runtime configuration
- Validates configuration and instantiates appropriate implementations
- Enables extensibility for future providers without modifying core code
- Separates configuration concerns from implementation details

**Key Design Attributes:**
- Registry pattern for known providers (embedding and vector store)
- Configuration-driven instantiation (no hardcoding of provider types)
- Centralized validation and error handling
- Discovery functions to list available provider options
- Extensible design pattern for adding new providers

**Requirements:**
- Support configuration-driven provider selection from environment variables
- Validate configuration before instantiation
- Provide clear error messages for unknown providers or invalid configuration
- Enable dynamic registration of new providers (for future extensibility)
- Return fully initialized provider instances implementing established protocols

### 11.4 End-to-End Data Flow

**Document Ingestion Pipeline:**
```
Document Text
    ↓
ChunkingService.chunk() [Phase 1]
    ↓ List[Chunk]
EmbeddingProvider.embed_batch() [Phase 2 - NEW]
    ↓ List[Vector] (1536 dims each)
VectorStore.insert() [Phase 2 - NEW]
    ↓
Indexed in Qdrant
    ↓
Searchable via semantic similarity
```

**Query Execution Pipeline:**
```
User Query
    ↓
EmbeddingProvider.embed_text() [Phase 2 - NEW]
    ↓ Query Vector (1536 dims)
VectorStore.search() [Phase 2 - NEW]
    ↓ List[SearchResult] (top-K similar chunks)
Format and Return
    ↓
Most Relevant Chunks (ready for LLM augmentation)
```

**Configuration Strategy:**
- Environment variable-driven configuration (no hardcoding)
- Separate configuration for each provider (embedding, vector store)
- Sensible defaults for development and production
- Required credentials sourced from secure environment variables
- Configuration validation at initialization time

**Key Configuration Concerns:**
- Provider type selection for embedding and vector storage
- Provider-specific parameters (models, endpoints, credentials, timeouts)
- Performance tuning (batch sizes, retry counts)
- Vector dimensionality alignment between embedding provider and vector store
- Distance metric selection for similarity search

## 12. Roadmap

### Phase 1 (Day 2) - ✅ COMPLETE
- Abstract protocols for EmbeddingProvider, VectorStore
- LangChain-based text chunking (recursive, markdown)
- Configuration-driven dependency injection
- 24 comprehensive unit tests
- Module documentation

### Phase 2a (Day 10) - ✅ COMPLETE
- ✅ OpenAI EmbeddingProvider implementation
- ✅ NVIDIA EmbeddingProvider implementation (Nemotron 3.5)
- ✅ Qdrant VectorStore implementation
- ✅ Factory pattern extension
- ✅ 30+ comprehensive unit and integration tests
- ✅ Configuration management (environment variables)
- ✅ End-to-end integration testing
- ✅ Documentation updates (RFC, SRS, ADR, C4, EDI)

### Phase 2b (Day 11) - 🔄 IN PROGRESS
- RAG Service Layer - Orchestration & API Endpoints
  - RAGService class (embedding + vector store orchestration)
  - Request/response Pydantic schemas
  - 6 REST API endpoints (health, embed, ingest, search, query, collections)
  - Dependency injection and provider wiring
  - 33+ integration tests
  - Complete API documentation
  - SRS and ADR updates

### Phase 3 (Future) - Planned
- RAG integration with agent service
- Retrieval in agent prompts
- RAG observability and metrics
- Performance benchmarking

### Phase 4+ - Future
- Additional embedding providers (Claude, local, HuggingFace)
- Additional vector stores (Weaviate, Pinecone, ChromaDB)
- Advanced chunking: code-aware, HTML, PDF, multi-language
- Hybrid retrieval (BM25 + vector)
- Re-ranking strategies
- RAG evaluation framework
- Cost tracking and FinOps

## 13. Success Criteria

### Phase 1 - ✅ COMPLETE
- [x] Abstract protocols (EmbeddingProvider, VectorStore) defined
- [x] 2+ chunking strategies implemented with LangChain
- [x] Factory pattern with configuration-driven selection
- [x] 24 unit tests covering all scenarios
- [x] Type checking (mypy) clean
- [x] Code quality (ruff) clean
- [x] Comprehensive module documentation

### Phase 2a - ✅ COMPLETE
- [x] OpenAI EmbeddingProvider fully implemented
- [x] NVIDIA EmbeddingProvider fully implemented (Nemotron 3.5)
- [x] Qdrant VectorStore fully implemented
- [x] Factory pattern extended for providers
- [x] 30+ comprehensive unit and integration tests
- [x] Configuration management (environment variables)
- [x] Type safety (mypy --strict clean for new code)
- [x] Code quality (ruff clean for new code)
- [x] RFC-009 Phase 2a documentation complete
- [x] SRS-001 requirements added (FR-014, FR-015, NFR-009)
- [x] ADR-010 & ADR-011 created (provider selections)
- [x] C4-001 updated with new containers
- [x] EDI-001 updated with traceability
- [x] Module README extended with examples

### Phase 2b - 🔄 IN PROGRESS
- [ ] RAGService class fully implemented (200+ lines)
- [ ] Pydantic schemas for all 6 endpoints (150+ lines)
- [ ] 6 FastAPI endpoints fully implemented
- [ ] Dependency injection and provider wiring
- [ ] 33+ integration tests all passing
- [ ] Configuration-driven provider selection (NVIDIA/OpenAI)
- [ ] Type safety (mypy --strict clean)
- [ ] Code quality (ruff clean)
- [ ] API documentation complete with examples
- [ ] SRS-001 updated with endpoint specifications
- [ ] ADR created for RAG Service wiring strategy
- [ ] No test regressions (203+ existing tests still pass)

### Phase 3+ - Future
- [ ] RAG integration with agent service
- [ ] Retrieval in agent prompts
- [ ] RAG observability and metrics

## 13. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| LangChain API changes | Breaking changes | Lock to stable version, version tests |
| Protocol limitations | Runtime errors if impl. incorrect | Comprehensive protocol docs, example implementations |
| Configuration sprawl | Hard to manage | YAML config files, validation, sensible defaults |
| Embedding cost | High inference cost | Batch operations, caching, rate limiting (Phase 2) |
| Vector DB availability | Data loss, downtime | Backup strategy, HA setup (Phase 2) |

## 14. References

- AEGIS-SRS-001 v1.7: FR-013 (RAG Functional), NFR-008 (RAG Non-Functional)
- AEGIS-ADR-009: RAG Chunking strategy decision
- AEGIS-C4-001 v1.7: RAG Retriever container
- LangChain Documentation: Text Splitters
- OpenAI Embeddings API
- Qdrant Vector Database

## 15. Appendix: Example Usage

### Chunking Example
```python
from aegis_platform.rag.chunking import create_chunker
from aegis_platform.rag.models import ChunkingConfig

# Create chunker with recursive strategy
config = ChunkingConfig(
    strategy="recursive",
    chunk_size=1024,
    overlap=128,
)
chunker = create_chunker(config)

# Chunk a document
document = """
# My Document

This is the first paragraph.

## Section 2

This is another paragraph with more content.
"""

## 15. Appendix: Design Patterns & Principles

### Protocol-Based Abstractions
The architecture uses `@runtime_checkable` protocols to define contracts for embedding providers and vector stores. This enables:
- Runtime duck-typing without inheritance hierarchies
- Easy mocking and testing with protocol-compliant fakes
- Loose coupling to specific vendor implementations
- Future providers can implement protocols without modifying core code

### Configuration-Driven Selection
All provider instantiation is configuration-driven via environment variables:
- No hardcoding of provider types or credentials
- Support for switching providers without code changes
- Clear validation errors for missing or invalid configuration
- Extensible patterns for future configuration management

### Factory Pattern for Provider Instantiation
A centralized factory abstracts provider creation:
- Single responsibility: validate configuration and instantiate providers
- Registry pattern: maintains known providers and enables registration of new ones
- Error handling: centralized validation logic
- Testability: easy to mock or stub for testing

### End-to-End Data Flow
Document ingestion and query execution follow predictable pipelines:
- **Ingestion:** Document → Chunking → Embedding → Vector Storage → Indexed
- **Query:** User Query → Embedding → Vector Search → Relevant Chunks → LLM Context

---

**Author:** AEGIS Platform Team
**Last Updated:** 2026-11-15
**Status:** Accepted for Phase 1 Implementation
