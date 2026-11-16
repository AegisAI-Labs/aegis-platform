"""RAG (Retrieval-Augmented Generation) module for AEGIS Platform.

This module provides abstract protocols and implementations for:
- Document embeddings with pluggable providers
- Configurable document chunking strategies (LangChain-based)
- Vector store interfaces for retrieval

All components are designed as abstract protocols to support dependency
injection and runtime provider selection via configuration files.

Package structure:
- embeddings.py: Abstract EmbeddingProvider protocol (no concrete implementations)
- chunking/: LangChain-based chunking strategies subpackage
  - strategy.py: Abstract ChunkingStrategy protocol and ChunkingConfig
  - langchain_recursive.py: LangChainRecursiveChunker (LangChain wrapper)
  - langchain_markdown.py: LangChainMarkdownChunker (LangChain wrapper)
  - factory.py: create_chunker() factory function
- storage.py: Abstract VectorStore protocol
- models.py: Data models (Chunk, Document, SearchResult, etc.)
"""

from aegis_platform.rag.chunking import (
    ChunkingConfig,
    ChunkingStrategy,
    create_chunker,
)
from aegis_platform.rag.embeddings import EmbeddingConfig, EmbeddingProvider
from aegis_platform.rag.models import (
    Chunk,
    Document,
    EmbeddingRequest,
    EmbeddingResponse,
    SearchResult,
)
from aegis_platform.rag.storage import VectorStore, VectorStoreConfig

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "ChunkingStrategy",
    "Document",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "SearchResult",
    "VectorStore",
    "VectorStoreConfig",
    "create_chunker",
]
