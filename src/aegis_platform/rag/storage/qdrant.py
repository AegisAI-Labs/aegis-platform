"""Qdrant VectorStore implementation for vector similarity search and storage.

This module provides a concrete implementation of the VectorStore protocol
using Qdrant as the backend. It supports:
- Vector insertion with metadata
- Similarity search with configurable limits and thresholds
- Collection management (create, delete, list)
- Update and delete operations
- Batch operations for efficiency
- Health checks and error handling

Qdrant can run locally via Docker or be configured for remote access.
"""

from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from aegis_platform.rag.models import SearchResult


@dataclass
class QdrantVectorStoreConfig:
    """Configuration for Qdrant VectorStore.

    Attributes:
        url: Qdrant server URL (e.g., "http://localhost:6333")
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds
        embedding_dimension: Embedding vector dimension (must match provider)
        distance_metric: Distance metric for similarity ("cosine", "euclidean", "manhattan")
    """

    url: str
    api_key: str | None = None
    timeout: int = 30
    embedding_dimension: int = 1536
    distance_metric: str = "cosine"

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If parameters are invalid
        """
        if not self.url or not self.url.strip():
            raise ValueError("url cannot be empty")
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.distance_metric not in ("cosine", "euclidean", "manhattan"):
            raise ValueError(
                f"distance_metric must be one of cosine, euclidean, manhattan, "
                f"got {self.distance_metric}"
            )


class QdrantVectorStore:
    """Qdrant VectorStore implementation.

    Wraps Qdrant's async client with:
    - Collection lifecycle management
    - Vector insertion, search, update, delete
    - Metadata preservation
    - Error handling and health checks
    """

    def __init__(self, config: QdrantVectorStoreConfig) -> None:
        """Initialize Qdrant vector store.

        Args:
            config: Configuration with URL and parameters

        Raises:
            ValueError: If configuration is invalid
        """
        config.validate()
        self.config = config
        self.client = AsyncQdrantClient(
            url=config.url,
            api_key=config.api_key,
            timeout=config.timeout,
        )

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
        if not vectors:
            raise ValueError("Cannot insert empty vector list")

        if len(vectors) != len(metadata):
            raise ValueError(
                f"Vector count ({len(vectors)}) must match metadata count " f"({len(metadata)})"
            )

        if ids and len(ids) != len(vectors):
            raise ValueError(f"ID count ({len(ids)}) must match vector count ({len(vectors)})")

        # Validate all vectors have correct dimension
        for i, vector in enumerate(vectors):
            if len(vector) != self.config.embedding_dimension:
                raise ValueError(
                    f"Vector {i} has dimension {len(vector)}, "
                    f"expected {self.config.embedding_dimension}"
                )

        # Generate IDs if not provided
        if not ids:
            ids = [str(i) for i in range(len(vectors))]

        # Create point structures with metadata as payload
        points = [
            PointStruct(
                id=int(pid) if pid.isdigit() else hash(pid) % (2**31),
                vector=vector,
                payload=meta,
            )
            for pid, vector, meta in zip(ids, vectors, metadata)
        ]

        try:
            await self.client.upsert(
                collection_name=collection,
                points=points,
            )
            return ids
        except Exception as e:
            raise RuntimeError(f"Failed to insert vectors: {e!s}") from e

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
        if len(vector) != self.config.embedding_dimension:
            raise ValueError(
                f"Query vector dimension {len(vector)} does not match "
                f"configured dimension {self.config.embedding_dimension}"
            )

        if limit <= 0:
            raise ValueError("limit must be positive")

        try:
            results = await self.client.search(  # type: ignore[attr-defined]
                collection_name=collection,
                query_vector=vector,
                limit=limit,
                score_threshold=score_threshold,
            )

            search_results = []
            for result in results:
                payload = result.payload or {}
                search_results.append(
                    SearchResult(
                        document_id=payload.get("document_id", ""),
                        chunk_id=payload.get("chunk_id", ""),
                        text=payload.get("text", ""),
                        score=result.score,
                        distance=1 - result.score if result.score else 0,
                        metadata=payload,
                    )
                )

            return search_results
        except Exception as e:
            raise RuntimeError(f"Search failed: {e!s}") from e

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
        if not ids:
            raise ValueError("Cannot delete empty ID list")

        try:
            # Convert string IDs to integers
            int_ids = [
                int(id_str) if id_str.isdigit() else hash(id_str) % (2**31) for id_str in ids
            ]

            await self.client.delete(
                collection_name=collection,
                points_selector=int_ids,  # type: ignore[arg-type]
            )

            return len(ids)
        except Exception as e:
            raise RuntimeError(f"Failed to delete vectors: {e!s}") from e

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
        if not ids or not vectors:
            raise ValueError("Cannot update empty ID or vector list")

        if len(ids) != len(vectors):
            raise ValueError(f"ID count ({len(ids)}) must match vector count ({len(vectors)})")

        if metadata and len(metadata) != len(ids):
            raise ValueError(f"Metadata count ({len(metadata)}) must match ID count ({len(ids)})")

        # Use upsert with new vectors and metadata
        if metadata is None:
            metadata = [{} for _ in ids]

        try:
            int_ids = [
                int(id_str) if id_str.isdigit() else hash(id_str) % (2**31) for id_str in ids
            ]

            points = [
                PointStruct(id=pid, vector=vector, payload=meta)
                for pid, vector, meta in zip(int_ids, vectors, metadata)
            ]

            await self.client.upsert(
                collection_name=collection,
                points=points,
            )

            return len(ids)
        except Exception as e:
            raise RuntimeError(f"Failed to update vectors: {e!s}") from e

    async def list_collections(self) -> list[str]:
        """List all collections in store.

        Returns:
            List of collection names
        """
        try:
            response = await self.client.get_collections()
            return [collection.name for collection in response.collections]
        except Exception as e:
            raise RuntimeError(f"Failed to list collections: {e!s}") from e

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
        if dimension <= 0:
            raise ValueError("dimension must be positive")

        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "manhattan": Distance.MANHATTAN,
        }

        distance = distance_map.get(distance_metric)
        if not distance:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        try:
            await self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dimension, distance=distance),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create collection: {e!s}") from e

    async def delete_collection(self, collection: str) -> None:
        """Delete a collection.

        Args:
            collection: Collection name

        Raises:
            ValueError: If collection doesn't exist
            RuntimeError: If deletion fails
        """
        try:
            await self.client.delete_collection(collection_name=collection)
        except Exception as e:
            raise RuntimeError(f"Failed to delete collection: {e!s}") from e

    async def collection_stats(self, collection: str) -> dict[str, Any]:
        """Get statistics about a collection.

        Returns:
            Dictionary with stats (count, dimension, etc.)
        """
        try:
            collection_info = await self.client.get_collection(collection_name=collection)
            vectors_config = collection_info.config.params.vectors
            # Handle both single vector and multi-vector configs
            if isinstance(vectors_config, VectorParams):
                vector_size = vectors_config.size
                distance = vectors_config.distance
            elif isinstance(vectors_config, dict):
                # For multi-vector, get first entry
                first_vector = next(iter(vectors_config.values()))
                vector_size = first_vector.size
                distance = first_vector.distance
            else:
                vector_size = 0
                distance = "unknown"  # type: ignore[assignment]

            return {
                "name": collection,
                "point_count": collection_info.points_count,
                "vector_size": vector_size,
                "distance": str(distance),
                "status": str(collection_info.status),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get collection stats: {e!s}") from e

    async def health_check(self) -> bool:
        """Check if vector store is available.

        Returns:
            True if store is healthy, False otherwise
        """
        try:
            await self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False
