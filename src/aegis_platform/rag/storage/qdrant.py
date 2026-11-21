"""Qdrant VectorStore implementation for vector similarity search and storage.

This module provides a concrete implementation of the VectorStore protocol
using Qdrant as the backend. It supports:
- Vector insertion with metadata
- Similarity search with configurable limits and thresholds
- Collection management (create, delete, list)
- Update and delete operations
- Batch operations for efficiency
- Health checks and error handling

Qdrant can run locally via Docker, remotely, or in-process with ``url=":memory:"``
(qdrant-client local mode; used for tests and local experiments).

Point IDs: Qdrant accepts unsigned integers or UUIDs. Caller IDs such as ``"doc-1#0"`` are
mapped to deterministic UUIDv5 values, so re-ingesting the same chunk overwrites it instead of
creating a duplicate, and the mapping is stable across processes and restarts.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HasIdCondition,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from aegis_platform.rag.models import SearchResult

logger = logging.getLogger(__name__)

IN_MEMORY_URL = ":memory:"

# Payload field identifying a chunk's document; indexed so per-document cleanup is a
# filtered lookup rather than a full scan (and works on clusters that require indexes).
DOCUMENT_ID_FIELD = "document_id"

# Fixed namespace so the same caller ID always maps to the same point ID.
# NEVER change this value: existing points would no longer match their chunk IDs, so
# delete/update by ID would silently miss them. It is deliberately not configurable;
# separate collections already have separate ID spaces (see ADR-012).
_POINT_ID_NAMESPACE = uuid.UUID("6f1c2a4e-3b8d-5e2f-9a7c-1d4b6e8f0a3c")

_DISTANCES = {
    "cosine": Distance.COSINE,
    "euclidean": Distance.EUCLID,
    "manhattan": Distance.MANHATTAN,
}


class CollectionMismatchError(RuntimeError):
    """The collection was built with a different embedding model, dimension or metric.

    Vectors from different models are not comparable, so mixing them silently ruins
    search quality. Use a new collection (RAG_COLLECTION) or recreate this one.
    """


def to_point_id(raw_id: str) -> int | str:
    """Map a caller ID to a stable Qdrant point ID.

    Plain non-negative integers are kept as integers; every other string becomes a
    deterministic UUIDv5. (Python's ``hash()`` was used before; it is randomized per
    process, so IDs changed after every restart.)
    """
    if raw_id.isdigit():
        return int(raw_id)
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, raw_id))


@dataclass
class QdrantVectorStoreConfig:
    """Configuration for Qdrant VectorStore.

    Attributes:
        url: Qdrant server URL (e.g., "http://localhost:6333")
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds
        embedding_dimension: Embedding vector dimension (must match provider)
        distance_metric: Distance metric for similarity ("cosine", "euclidean", "manhattan")
        embedding_model: Embedding model that produces the vectors. Recorded in the metadata
            of collections this store creates and checked against existing collections.
    """

    url: str
    api_key: str | None = None
    timeout: int = 30
    embedding_dimension: int = 1536
    distance_metric: str = "cosine"
    embedding_model: str | None = None

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
        if config.url == IN_MEMORY_URL:
            self.client = AsyncQdrantClient(location=IN_MEMORY_URL)
        else:
            self.client = AsyncQdrantClient(
                url=config.url,
                api_key=config.api_key,
                timeout=config.timeout,
            )
        self._known_collections: set[str] = set()

    def _collection_metadata(self, dimension: int, distance_metric: str) -> dict[str, Any]:
        return {
            "embedding_model": self.config.embedding_model,
            "embedding_dimension": dimension,
            "distance_metric": distance_metric,
        }

    @staticmethod
    def _describe(model: Any, dimension: Any, distance: Any) -> str:
        return f"{model or 'unknown model'} (dimension {dimension}, {distance})"

    async def verify_collection(self, collection: str) -> bool:
        """Check that ``collection`` matches this store's embedding configuration.

        Returns:
            False if the collection does not exist, True if it exists and matches.

        Raises:
            CollectionMismatchError: If it was built with a different model, dimension or
                distance metric.
            RuntimeError: If the collection cannot be inspected.
        """
        if collection in self._known_collections:
            return True

        try:
            if not await self.client.collection_exists(collection_name=collection):
                return False
            stats = await self.collection_stats(collection)
            info = await self.client.get_collection(collection_name=collection)
        except Exception as e:
            raise RuntimeError(f"Failed to inspect collection {collection!r}: {e!s}") from e

        metadata = getattr(info.config, "metadata", None) or {}
        stored_model = metadata.get("embedding_model")
        expected_distance = str(_DISTANCES[self.config.distance_metric].value).lower()

        model_differs = bool(
            stored_model
            and self.config.embedding_model
            and stored_model != self.config.embedding_model
        )
        if (
            stats["vector_size"] != self.config.embedding_dimension
            or stats["distance"] != expected_distance
            or model_differs
        ):
            built = self._describe(stored_model, stats["vector_size"], stats["distance"])
            current = self._describe(
                self.config.embedding_model, self.config.embedding_dimension, expected_distance
            )
            raise CollectionMismatchError(
                f"Collection {collection!r} was built with {built}; the current configuration "
                f"is {current}. Set RAG_COLLECTION to a new collection and re-ingest, "
                "or recreate this one."
            )

        if not stored_model:
            logger.warning(
                "Collection %r has no recorded embedding model; only dimension and metric "
                "were verified",
                collection,
            )

        self._known_collections.add(collection)
        return True

    async def ensure_collection(self, collection: str) -> bool:
        """Create ``collection`` for this store's configuration, or verify the existing one.

        Returns:
            True if the collection was created, False if it already existed (and matches).

        Raises:
            CollectionMismatchError: If an existing collection does not match.
        """
        if await self.verify_collection(collection):
            return False

        try:
            await self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=self.config.embedding_dimension,
                    distance=_DISTANCES[self.config.distance_metric],
                ),
                metadata=self._collection_metadata(
                    self.config.embedding_dimension, self.config.distance_metric
                ),
            )
            await self.client.create_payload_index(
                collection_name=collection,
                field_name=DOCUMENT_ID_FIELD,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to ensure collection {collection!r}: {e!s}") from e

        self._known_collections.add(collection)
        return True

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
                f"Vector count ({len(vectors)}) must match metadata count ({len(metadata)})"
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
                id=to_point_id(pid),
                vector=vector,
                payload=meta,
            )
            for pid, vector, meta in zip(ids, vectors, metadata, strict=True)
        ]

        await self.ensure_collection(collection)

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

        if not await self.verify_collection(collection):
            # Nothing has been ingested yet: an empty result, not an error.
            return []

        try:
            response = await self.client.query_points(
                collection_name=collection,
                query=vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

            search_results = []
            for result in response.points:
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
            await self.client.delete(
                collection_name=collection,
                points_selector=PointIdsList(points=[to_point_id(id_str) for id_str in ids]),
            )

            return len(ids)
        except Exception as e:
            raise RuntimeError(f"Failed to delete vectors: {e!s}") from e

    async def delete_stale_chunks(
        self, collection: str, document_id: str, keep_ids: list[str]
    ) -> int:
        """Delete ``document_id``'s points whose IDs are not in ``keep_ids``.

        Run after upserting the current version, so the document is never missing from search;
        also removes points left from before deterministic IDs (ADR-012).
        """
        if not document_id:
            raise ValueError("document_id is required")

        stale = Filter(
            must=[FieldCondition(key=DOCUMENT_ID_FIELD, match=MatchValue(value=document_id))],
            must_not=[HasIdCondition(has_id=[to_point_id(i) for i in keep_ids])]
            if keep_ids
            else [],
        )

        if not await self.verify_collection(collection):
            return 0

        try:
            count = (
                await self.client.count(collection_name=collection, count_filter=stale, exact=True)
            ).count
            if count:
                await self.client.delete(
                    collection_name=collection,
                    points_selector=FilterSelector(filter=stale),
                )
            return count
        except Exception as e:
            raise RuntimeError(f"Failed to delete stale chunks: {e!s}") from e

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
            points = [
                PointStruct(id=to_point_id(pid), vector=vector, payload=meta)
                for pid, vector, meta in zip(ids, vectors, metadata, strict=True)
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

        distance = _DISTANCES.get(distance_metric)
        if not distance:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        try:
            exists = await self.client.collection_exists(collection_name=collection)
        except Exception as e:
            raise RuntimeError(f"Failed to create collection: {e!s}") from e

        if exists:
            # Never drop data implicitly (the previous recreate_collection call deleted it).
            raise ValueError(f"Collection already exists: {collection}")

        try:
            await self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dimension, distance=distance),
                metadata=self._collection_metadata(dimension, distance_metric),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create collection: {e!s}") from e

        self._known_collections.add(collection)

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
        finally:
            self._known_collections.discard(collection)

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
                "point_count": collection_info.points_count or 0,
                "vector_size": vector_size,
                "distance": str(getattr(distance, "value", distance)).lower(),
                "status": str(getattr(collection_info.status, "value", collection_info.status)),
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
