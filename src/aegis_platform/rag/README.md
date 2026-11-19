# RAG (Retrieval-Augmented Generation) Module

> **Production-Grade RAG Foundation for AEGIS AI Platform**

## Overview

The RAG module provides a provider-agnostic, production-ready foundation for building retrieval-augmented generation pipelines. It decouples document chunking, embeddings, and vector storage through abstract protocols and dependency injection, enabling runtime strategy selection via configuration.

## Architecture

The RAG module follows a **layered protocol-based architecture**:

```
┌─────────────────────────────────────────────────────────┐
│         Application Layer (Agent, API)                  │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼──────────┐    ┌────────▼─────────┐
│  Document Loader │    │  Query Handler   │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
┌────────▼──────────────────────▼──────────┐
│         RAG Orchestration Layer          │
├──────────────────────────────────────────┤
│ - Indexing Pipeline                      │
│ - Retrieval Pipeline                     │
│ - Reranking / Fusion                     │
└────────────┬──────────────────┬──────────┘
             │                  │
    ┌────────▼──────┐  ┌───────▼────────┐
    │   Chunking    │  │ Embeddings &   │
    │   Strategies  │  │ Vector Storage │
    └───────┬───────┘  └────────┬───────┘
            │                   │
    ┌───────▼─────────┐  ┌──────▼──────────┐
    │ ChunkingStrategy│  │ EmbeddingProvider
    │ Protocol        │  │ VectorStore
    │                 │  │ Protocols
    │ - Recursive     │  │ (Abstract)
    │ - Markdown      │  │
    │ - (Custom)      │  │ Concrete:
    └─────────────────┘  │ - (Planned)
                         └─────────────────┘
```

## Module Structure

### Core Components

```
src/aegis_platform/rag/
├── __init__.py              # Public API exports
├── models.py                # Data models (Chunk, Document, SearchResult, etc.)
├── embeddings.py            # EmbeddingProvider protocol (abstract)
├── storage.py               # VectorStore protocol (abstract)
└── chunking/
    ├── __init__.py          # Chunking subpackage exports
    ├── strategy.py          # ChunkingStrategy protocol + ChunkingConfig
    ├── factory.py           # create_chunker() factory function
    ├── langchain_recursive.py   # LangChainRecursiveChunker implementation
    └── langchain_markdown.py    # LangChainMarkdownChunker implementation
```

### Test Structure

```
tests/unit/rag/
└── test_chunking.py         # Unit tests for chunking strategies (24 tests)
```

## Data Models

### Chunk
Represents a unit of text extracted from a document.

```python
from aegis_platform.rag import Chunk

chunk = Chunk(
    text="The quick brown fox...",
    chunk_id="doc-123#0",
    sequence=0,
    start_pos=0,
    end_pos=21,
    metadata={
        "strategy": "langchain-recursive",
        "document_id": "doc-123",
        "source": "article.md"
    }
)
```

### Document
Represents a source document to be chunked and embedded.

```python
from aegis_platform.rag import Document

doc = Document(
    document_id="doc-123",
    content="Long text content...",
    metadata={"source": "article.md", "author": "John Doe"}
)
```

### SearchResult
Represents a result from vector similarity search.

```python
from aegis_platform.rag import SearchResult

result = SearchResult(
    chunk_id="doc-123#0",
    text="The quick brown fox...",
    score=0.92,
    metadata={...}
)
```

## Chunking Strategies

The chunking module provides LangChain-based text splitting strategies optimized for different content types.

### Available Strategies

#### 1. **Recursive** (`recursive`)
Handles varied text structures by recursively splitting on separators in order of preference.

```python
from aegis_platform.rag import ChunkingConfig, create_chunker

config = ChunkingConfig(
    strategy="recursive",
    chunk_size=512,
    overlap=50
)
chunker = create_chunker(config)

# Chunks text handling paragraphs, lines, words, characters in order
chunks = chunker.chunk("Long document text...", document_id="doc-1")
```

**Best for:**
- General-purpose documents (articles, reports, books)
- Mixed content (paragraphs, lists, code snippets)
- Preserving logical document structure

**Implementation:** LangChain's `RecursiveCharacterTextSplitter`
**Separator hierarchy:** `\n\n` → `\n` → ` ` → `` (empty string)

#### 2. **Markdown** (`markdown`)
Markdown-aware splitting with code block preservation.

```python
from aegis_platform.rag import ChunkingConfig, create_chunker

config = ChunkingConfig(
    strategy="markdown",
    chunk_size=1024,
    overlap=100
)
chunker = create_chunker(config)

# Chunks markdown while preserving headers and code blocks
chunks = chunker.chunk("# Title\n\n```python\ncode\n```\nContent...",
                       document_id="doc-2")
```

**Best for:**
- Markdown files (documentation, README files)
- Technical content with code examples
- Content with hierarchical headers

**Implementation:** LangChain's `MarkdownTextSplitter`
**Features:**
- Preserves markdown headers (`#`, `##`, etc.)
- Keeps code blocks intact
- Respects list structure

### Configuration

All chunkers support runtime configuration via `ChunkingConfig`:

```python
from aegis_platform.rag import ChunkingConfig

config = ChunkingConfig(
    strategy="recursive",      # Strategy name
    chunk_size=512,            # Target chunk size in characters
    overlap=50,                # Overlap between consecutive chunks
    min_chunk_size=50,         # Minimum chunk size (smaller chunks discarded)
    max_chunk_size=2000        # Maximum chunk size (chunks split further if needed)
)
```

### Runtime Configuration Update

Change chunking strategy at runtime:

```python
chunker = create_chunker(config)

# Initial chunking
chunks_v1 = chunker.chunk(text, "doc-1")

# Update configuration
chunker.update_config(chunk_size=256)

# Chunks will now use new size
chunks_v2 = chunker.chunk(text, "doc-2")
```

This is useful for:
- A/B testing different chunk sizes
- Adapting to different document types
- Experiments on retrieval performance

### Factory Function

Create chunkers via the factory for dependency injection:

```python
from aegis_platform.rag.chunking import (
    ChunkingConfig,
    create_chunker,
    get_available_strategies
)

# List available strategies
strategies = get_available_strategies()
print(strategies)  # ["recursive", "markdown"]

# Create chunker from config
config = ChunkingConfig(strategy="recursive", chunk_size=512)
chunker = create_chunker(config)

# Use chunker
chunks = chunker.chunk("Text to chunk...", "doc-id")
```

## Embedding Provider (Abstract Protocol)

The `EmbeddingProvider` protocol defines the interface for text embedding services.

```python
from aegis_platform.rag import EmbeddingProvider, EmbeddingRequest, EmbeddingResponse

class CustomEmbedder(EmbeddingProvider):
    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Convert text to embeddings."""
        # Implementation: call embedding API, return vectors
        pass
```

**Status:** Abstract protocol only (no concrete implementations yet)
**Planned implementations:** OpenAI, Claude, Llama, Local models

## Vector Store (Abstract Protocol)

The `VectorStore` protocol defines the interface for vector similarity search and storage.

```python
from aegis_platform.rag import VectorStore, SearchResult

class CustomVectorStore(VectorStore):
    def index(self, chunks: list[Chunk]) -> None:
        """Index chunks for similarity search."""
        pass

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        """Search for similar chunks."""
        pass
```

**Status:** Abstract protocol only (no concrete implementations yet)
**Planned implementations:** Qdrant, Weaviate, Pinecone, Chroma

## Usage Examples

### Example 1: Basic Document Chunking

```python
from aegis_platform.rag import ChunkingConfig, create_chunker, Document

# Create chunker
config = ChunkingConfig(strategy="recursive", chunk_size=512, overlap=50)
chunker = create_chunker(config)

# Chunk a document
document = Document(
    document_id="research-paper-1",
    content="Long research paper content...",
    metadata={"title": "AI Safety", "year": 2024}
)

chunks = chunker.chunk(document.content, document.document_id)

# Process chunks
for chunk in chunks:
    print(f"Chunk {chunk.sequence}: {chunk.text[:50]}...")
    print(f"  Metadata: {chunk.metadata}")
```

### Example 2: Multi-Strategy Chunking (A/B Testing)

```python
from aegis_platform.rag import ChunkingConfig, create_chunker

text = "Complex document with code and markdown..."

# Strategy A: Recursive
config_a = ChunkingConfig(strategy="recursive", chunk_size=512)
chunker_a = create_chunker(config_a)
chunks_a = chunker_a.chunk(text, "doc-1")

# Strategy B: Markdown
config_b = ChunkingConfig(strategy="markdown", chunk_size=1024)
chunker_b = create_chunker(config_b)
chunks_b = chunker_b.chunk(text, "doc-1")

# Compare strategies
print(f"Recursive: {len(chunks_a)} chunks")
print(f"Markdown: {len(chunks_b)} chunks")
```

### Example 3: Dynamic Configuration Update

```python
from aegis_platform.rag import ChunkingConfig, create_chunker

chunker = create_chunker(ChunkingConfig(strategy="recursive", chunk_size=512))

# Experiment with different sizes
for chunk_size in [256, 512, 1024, 2048]:
    chunker.update_config(chunk_size=chunk_size)
    chunks = chunker.chunk(text, f"doc-{chunk_size}")
    print(f"Size {chunk_size}: {len(chunks)} chunks")
```

## Configuration via Environment Variables

Configure RAG settings in `.env`:

```dotenv
# Chunking Strategy
RAG_CHUNKING_STRATEGY=recursive
RAG_CHUNKING_CHUNK_SIZE=512
RAG_CHUNKING_OVERLAP=50
RAG_CHUNKING_MIN_CHUNK_SIZE=50
RAG_CHUNKING_MAX_CHUNK_SIZE=2000

# Embedding Provider (TBD)
RAG_EMBEDDING_PROVIDER=openai
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_EMBEDDING_DIMENSION=1536

# Vector Store (TBD)
RAG_VECTOR_STORE=qdrant
RAG_VECTOR_STORE_URL=http://localhost:6333
```

## Testing

### Run Unit Tests

```bash
# Run all RAG tests
uv run pytest tests/unit/rag/ -v

# Run only chunking tests
uv run pytest tests/unit/rag/test_chunking.py -v

# Run with coverage
uv run pytest tests/unit/rag/ --cov=aegis_platform.rag
```

### Test Coverage

- **Chunking tests:** 24 tests covering initialization, validation, chunking, configuration, metadata tracking, integration scenarios

## Type Checking & Linting

```bash
# Type checking
uv run mypy src/aegis_platform/rag/

# Linting
uv run ruff check src/aegis_platform/rag/
```

## Production Readiness

### Current Status (Day 2)

| Component | Status | Details |
|-----------|--------|---------|
| Chunking strategies | ✅ Production-ready | 2 LangChain implementations |
| Abstract protocols | ✅ Production-ready | EmbeddingProvider, VectorStore |
| Data models | ✅ Production-ready | Chunk, Document, SearchResult |
| Factory pattern | ✅ Production-ready | Config-driven DI |
| Unit tests | ✅ Complete | 24 tests, all passing |
| Type checking | ✅ Pass | mypy clean |
| Linting | ✅ Pass | ruff clean |
| Documentation | ✅ Complete | This README |

### Roadmap

**Day 3:** EmbeddingProvider & VectorStore implementations
- Implement OpenAI embeddings
- Implement Qdrant vector store
- Build integration tests

**Day 4:** Document loader & indexing pipeline
- Document loader for various file types
- Batch indexing pipeline
- Progress tracking

**Day 5:** Retrieval pipeline & reranking
- Query embedding
- Similarity search
- Result reranking (optional)

## Safety Layers

The module also provides safety layers that are composed around retrieval:

| Layer | Module |
|---|---|
| Query guardrails | `query_guardrails.py` |
| Query enhancement | `llm_query_enhancer.py`, `query_enhancer_base.py` |
| Document access control (RBAC, multi-tenant) | `access_control.py` |
| Cross-encoder reranking | `cross_encoder_reranker.py`, `reranker_base.py` |
| Citation tracking | `citation_tracker.py` |
| Hallucination detection | `hallucination_detector.py` |
| PII detection and masking | `pii_detector.py` |

See the [API reference](../../../docs/API/AEGIS-API-RAG.md),
[configuration guide](../../../docs/GUIDES/RAG-SAFETY-CONFIGURATION.md) and
[usage guide](../../../docs/GUIDES/RAG-SAFETY-USAGE.md).

## References

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_loaders/split_documents)
- [Vector Search Best Practices](https://qdrant.tech/documentation/cloud-quickstart/)
- [RAG Architecture Patterns](https://arxiv.org/abs/2312.10997)

## Contributing

When adding new chunking strategies:

1. Create a new file: `src/aegis_platform/rag/chunking/my_strategy.py`
2. Implement the `ChunkingStrategy` protocol
3. Register in `factory.py` CHUNKING_STRATEGIES dict
4. Add tests to `tests/unit/rag/test_chunking.py`
5. Update this README with strategy details

Example:

```python
# src/aegis_platform/rag/chunking/my_strategy.py
from aegis_platform.rag.chunking.strategy import ChunkingStrategy

class MyChunker(ChunkingStrategy):
    def chunk(self, text: str, document_id: str) -> list[Chunk]:
        # Your implementation
        pass

    def get_config(self) -> dict:
        # Return current config
        pass

    def update_config(self, **kwargs) -> None:
        # Update config
        pass
```

## License

MIT License
