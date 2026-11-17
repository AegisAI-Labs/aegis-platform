"""RAG (Retrieval-Augmented Generation) module for AEGIS Platform.

This module provides abstract protocols and implementations for:
- Document embeddings with pluggable providers (OpenAI, NVIDIA NIM)
- Configurable document chunking strategies (LangChain-based)
- Vector store interfaces for retrieval (Qdrant)

All components are designed as abstract protocols to support dependency
injection and runtime provider selection via configuration files.

Package structure:
- embeddings/: Embedding providers subpackage
  - openai.py: OpenAI embedding provider implementation
  - nvidia.py: NVIDIA NIM API embedding provider implementation
- storage/: Vector storage providers subpackage
  - qdrant.py: Qdrant vector database implementation
- chunking/: LangChain-based text chunking
- models.py: Data models (Chunk, Document, SearchResult, etc.)
- provider_factory.py: Factory functions for creating providers
"""

from aegis_platform.rag.chunking import (
    ChunkingConfig,
    ChunkingStrategy,
    create_chunker,
)
from aegis_platform.rag.embeddings import (
    EmbeddingProvider,
    NVIDIAEmbeddingConfig,
    NVIDIAEmbeddingProvider,
    OpenAIEmbedding,
    OpenAIEmbeddingConfig,
)
from aegis_platform.rag.models import (
    Chunk,
    Document,
    EmbeddingRequest,
    EmbeddingResponse,
    SearchResult,
)
from aegis_platform.rag.provider_factory import (
    create_embedding_provider,
    create_vector_store,
    get_available_embedding_providers,
    get_available_vector_stores,
)
from aegis_platform.rag.storage import (
    QdrantVectorStore,
    QdrantVectorStoreConfig,
    VectorStore,
    VectorStoreConfig,
)

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "ChunkingStrategy",
    "Document",
    "EmbeddingProvider",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "NVIDIAEmbeddingConfig",
    "NVIDIAEmbeddingProvider",
    "OpenAIEmbedding",
    "OpenAIEmbeddingConfig",
    "QdrantVectorStore",
    "QdrantVectorStoreConfig",
    "SearchResult",
    "VectorStore",
    "VectorStoreConfig",
    "create_chunker",
    "create_embedding_provider",
    "create_vector_store",
    "get_available_embedding_providers",
    "get_available_vector_stores",
]
