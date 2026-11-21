"""Vector storage providers for semantic search.

Modules:
    base: VectorStore protocol and VectorStoreConfig (single source of truth)
    qdrant: Qdrant vector database implementation

Usage:
    from aegis_platform.rag.storage import VectorStore, VectorStoreConfig, SearchResult
    from aegis_platform.rag.storage import QdrantVectorStore, QdrantVectorStoreConfig
"""

from aegis_platform.rag.models import SearchResult
from aegis_platform.rag.storage.base import VectorStore, VectorStoreConfig
from aegis_platform.rag.storage.qdrant import (
    QdrantVectorStore,
    QdrantVectorStoreConfig,
)

__all__ = [
    "QdrantVectorStore",
    "QdrantVectorStoreConfig",
    "SearchResult",
    "VectorStore",
    "VectorStoreConfig",
]
