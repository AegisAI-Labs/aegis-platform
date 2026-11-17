"""Vector storage providers for semantic search.

This package contains abstract protocols, configuration classes, and concrete
implementations for storing and retrieving embeddings using various vector databases.

Modules:
    base: Abstract VectorStore protocol and SearchResult model (deprecated - use __init__ below)
    qdrant: Qdrant vector database implementation

Classes exported:
    VectorStoreConfig: Configuration for vector store instantiation
    VectorStore: Abstract protocol for vector storage
    SearchResult: Result from vector store search
    QdrantVectorStore: Qdrant implementation
    QdrantVectorStoreConfig: Qdrant-specific configuration

Usage:
    from aegis_platform.rag.storage import VectorStore, VectorStoreConfig, SearchResult
    from aegis_platform.rag.storage.qdrant import QdrantVectorStore
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from aegis_platform.rag.models import SearchResult
from aegis_platform.rag.storage.qdrant import (
    QdrantVectorStore,
    QdrantVectorStoreConfig,
)


@dataclass
class VectorStoreConfig:
    """Configuration for vector store instantiation.

    Attributes:
        store_type: Store type (e.g., 'qdrant', 'weaviate', 'pinecone', 'memory')
        url: Connection URL if applicable
        api_key: API key if applicable
        dimension: Embedding dimension (must match embedding provider)
        distance_metric: Distance metric (e.g., 'cosine', 'euclidean', 'manhattan')
        timeout: Request timeout in seconds
    """

    store_type: str
    url: str | None = None
    api_key: str | None = None
    dimension: int = 0
    distance_metric: str = "cosine"
    timeout: int = 30


@runtime_checkable
class VectorStore(Protocol):
    """Abstract protocol for vector storage and retrieval.

    Implementations provide:
    - insert(): Store embeddings with metadata
    - search(): Find similar embeddings
    - delete(): Remove embeddings
    - update(): Modify embeddings
    - list_collections(): Discover stored collections
    - health_check(): Verify availability

    This is intentionally abstract to support multiple backends:
    - Qdrant (open-source, self-hosted)
    - Weaviate
    - Pinecone (serverless)
    - LanceDB
    - In-memory (for testing)
    - Custom enterprise solutions

    Stores are instantiated via factory function with VectorStoreConfig.
    """

    async def insert(
        self,
        collection: str,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Insert vectors into collection.

        Args:
            collection: Collection name
            vectors: List of embedding vectors
            metadata: List of metadata dicts (must match vector count)
            ids: Optional IDs; generated if not provided

        Returns:
            List of vector IDs (inserted)

        Raises:
            ValueError: If inputs are mismatched or invalid
            RuntimeError: If insertion fails
        """
        ...

    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            collection: Collection name
            vector: Query embedding vector
            limit: Maximum results to return
            score_threshold: Minimum similarity score

        Returns:
            List of SearchResult objects ranked by similarity

        Raises:
            ValueError: If collection doesn't exist or vector dimension mismatches
            RuntimeError: If search fails
        """
        ...

    async def delete(self, collection: str, ids: list[str]) -> int:
        """Delete vectors by ID.

        Args:
            collection: Collection name
            ids: List of vector IDs to delete

        Returns:
            Number of vectors deleted

        Raises:
            ValueError: If collection doesn't exist
            RuntimeError: If deletion fails
        """
        ...

    async def update(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Update vectors and/or metadata.

        Args:
            collection: Collection name
            ids: List of vector IDs to update
            vectors: New embedding vectors
            metadata: New metadata (optional)

        Returns:
            Number of vectors updated

        Raises:
            ValueError: If IDs or inputs are invalid
            RuntimeError: If update fails
        """
        ...

    async def list_collections(self) -> list[str]:
        """List all collections in store.

        Returns:
            List of collection names
        """
        ...

    async def create_collection(
        self, collection: str, dimension: int, distance_metric: str = "cosine"
    ) -> None:
        """Create a new collection.

        Args:
            collection: Collection name
            dimension: Vector dimension
            distance_metric: Distance metric to use

        Raises:
            ValueError: If collection already exists
            RuntimeError: If creation fails
        """
        ...

    async def delete_collection(self, collection: str) -> None:
        """Delete a collection.

        Args:
            collection: Collection name

        Raises:
            ValueError: If collection doesn't exist
            RuntimeError: If deletion fails
        """
        ...

    async def collection_stats(self, collection: str) -> dict[str, Any]:
        """Get statistics about a collection.

        Returns:
            Dictionary with stats (count, dimension, etc.)
        """
        ...

    async def health_check(self) -> bool:
        """Check if vector store is available.

        Returns:
            True if store is healthy, False otherwise
        """
        ...


__all__ = [
    "QdrantVectorStore",
    "QdrantVectorStoreConfig",
    "SearchResult",
    "VectorStore",
    "VectorStoreConfig",
]
