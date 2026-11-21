"""RAG Service orchestration layer for embedding and vector store operations.

This module provides the RAGService class which orchestrates:
- Text embedding (via configurable EmbeddingProvider)
- Document ingestion with replace semantics (chunk, embed as passages, upsert, remove stale chunks)
- Semantic search (embed as query, vector similarity)
- Retrieval for RAG (passages only; answer generation belongs to the agent layer)

The service is provider-agnostic, supporting OpenAI, NVIDIA, and future
embedding providers through the EmbeddingProvider protocol. All data lives in one
collection per service (``RAG_COLLECTION``, default ``DEFAULT_COLLECTION``).
See ADR-012 (retrieval path) and ADR-013 (embedding usage).
"""

import logging
import re
import time

import numpy as np

from aegis_platform.rag.chunking import create_chunker
from aegis_platform.rag.chunking.strategy import ChunkingConfig
from aegis_platform.rag.models import (
    PASSAGE,
    QUERY,
    Document,
    EmbeddingRequest,
    EmbeddingResponse,
)
from aegis_platform.rag.protocols import EmbeddingProvider, VectorStore

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "documents"

# Qdrant-safe collection names (letters, digits, "_", "-"; up to 255 characters).
_COLLECTION_NAME = re.compile(r"^[A-Za-z0-9_-]{1,255}$")


class RAGService:
    """
    Orchestration service for RAG operations.

    Coordinates embedding providers and vector stores to support:
    - Text embedding generation
    - Document ingestion and chunking (re-ingesting a document replaces it)
    - Semantic search
    - Retrieval of passages for RAG (no answer generation)

    Attributes:
        embedding_provider: Configured embedding provider (OpenAI, NVIDIA, etc.)
        vector_store: Configured vector store (Qdrant, etc.)
        collection: Vector store collection used for all operations (RAG_COLLECTION).
            Use a different collection per embedding model: vectors from different
            models are not comparable.
        chunker: Text chunking strategy
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        collection: str = DEFAULT_COLLECTION,
    ):
        """
        Initialize RAG Service with configured providers.

        Args:
            embedding_provider: EmbeddingProvider instance (OpenAI, NVIDIA, etc.)
            vector_store: VectorStore instance (Qdrant, etc.)
            collection: Collection name (default "documents")

        Raises:
            ValueError: If the collection name is invalid
        """
        if not _COLLECTION_NAME.fullmatch(collection):
            raise ValueError(
                f"Invalid collection name {collection!r}: use letters, digits, '_' or '-'"
            )
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.collection = collection

        # Initialize chunker with default configuration
        chunking_config = ChunkingConfig(
            strategy="recursive",
            chunk_size=1024,
            overlap=128,
            min_chunk_size=50,
            max_chunk_size=10000,
        )
        self.chunker = create_chunker(chunking_config)

        logger.info(
            "RAGService initialized with provider=%s, vector_store=%s, collection=%s",
            type(embedding_provider).__name__,
            type(vector_store).__name__,
            collection,
        )

    async def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text input (embedded as a search query).

        Args:
            text: Text to embed

        Returns:
            numpy array of embedding (dimension varies by provider)

        Raises:
            ValueError: If text is empty or too long
            RuntimeError: If embedding provider is unreachable
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if len(text) > 100000:
            raise ValueError("Text exceeds maximum length of 100,000 characters")

        logger.debug("Embedding text of length %d", len(text))

        try:
            embedding = await self.embedding_provider.embed_text(text)
            logger.debug("Successfully embedded text, dimension=%d", len(embedding))
            return embedding
        except Exception as e:
            logger.error("Failed to embed text: %s", e)
            raise RuntimeError(f"Embedding provider error: {e}") from e

    async def ingest_documents(
        self,
        documents: list[Document],
    ) -> dict:
        """
        Ingest documents: chunk, embed, upsert, then remove stale chunks.

        Re-ingesting a document **replaces** it (ADR-012):

        1. Chunk every document. Chunk IDs are ``"<document id>#<position>"`` and
           ``doc.metadata`` (including access metadata) is copied onto each chunk.
        2. Embed all chunks in one ``embed()`` call with ``input_type="passage"``.
           The provider must return exactly one vector per chunk.
        3. Upsert the chunks into ``self.collection``. Point IDs are deterministic,
           so chunks that already exist are overwritten, not duplicated. The collection is
           created on first use.
        4. For each document, delete its points that are not part of the new version
           (e.g. the tail of a document that got shorter, or points written with the
           pre-Day-16 random IDs).

        Step 4 runs only after step 3 succeeds, so a failed ingest leaves the previous
        version searchable. Each stored point's payload contains ``text``, ``chunk_id``,
        ``document_id`` (and legacy ``doc_id``), ``sequence`` and ``metadata``.

        Args:
            documents: List of Document objects to ingest

        Returns:
            Dictionary with:
            - chunks_created: Number of chunks produced (and stored) in this call
            - documents_stored: Number of documents processed
            - latency_ms: Total operation time
            - embedding_tokens: Tokens used for embedding, or None if the provider
              did not report usage (ADR-013)
            - chunks_removed: Stale chunks deleted from previous versions of these documents

        Raises:
            ValueError: If document list is empty
            RuntimeError: If chunking, embedding (including a vector/chunk count mismatch),
                storage or stale-chunk cleanup fails
        """
        if not documents:
            raise ValueError("Document list cannot be empty")

        start_time = time.time()

        try:
            all_chunks = []

            # Chunk all documents
            for doc in documents:
                logger.debug("Chunking document %s", doc.id)
                chunks = self.chunker.chunk(doc.content, doc.id)

                # Enrich chunks with document metadata
                for chunk in chunks:
                    if not chunk.metadata:
                        chunk.metadata = {}
                    chunk.metadata.update(doc.metadata or {})

                all_chunks.extend(chunks)

            logger.info("Created %d chunks from %d documents", len(all_chunks), len(documents))

            # Embed all chunks
            chunk_texts = [chunk.text for chunk in all_chunks]
            embedded = await self.embedding_provider.embed(
                EmbeddingRequest(texts=chunk_texts, input_type=PASSAGE)
            )
            embeddings = embedded.embeddings

            logger.info("Generated embeddings for %d chunks", len(embeddings))

            # Prepare vectors for storage
            vector_ids = []
            vector_data = []
            metadata_list = []

            if len(embeddings) != len(all_chunks):
                raise RuntimeError(
                    f"Embedding provider returned {len(embeddings)} vectors "
                    f"for {len(all_chunks)} chunks"
                )

            for chunk, embedding in zip(all_chunks, embeddings, strict=True):
                vector_ids.append(chunk.chunk_id)
                vector_data.append(
                    embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                )
                metadata_list.append(
                    {
                        "text": chunk.text,
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.doc_id,
                        "doc_id": chunk.doc_id,
                        "sequence": chunk.sequence,
                        "metadata": chunk.metadata or {},
                    }
                )

            # Store vectors in vector database
            await self.vector_store.insert(
                collection=self.collection,
                vectors=vector_data,
                metadata=metadata_list,
                ids=vector_ids,
            )

            # Replace semantics: drop chunks from an older version of each document
            # (e.g. a document that got shorter). Runs after the upsert, so a failed
            # ingest never leaves a document missing.
            chunks_removed = 0
            for doc in documents:
                chunks_removed += await self.vector_store.delete_stale_chunks(
                    collection=self.collection,
                    document_id=doc.id,
                    keep_ids=[chunk.chunk_id for chunk in all_chunks if chunk.doc_id == doc.id],
                )

            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                "Ingestion complete: chunks=%d, removed=%d, documents=%d, latency_ms=%.1f",
                len(all_chunks),
                chunks_removed,
                len(documents),
                latency_ms,
            )

            return {
                "chunks_created": len(all_chunks),
                "documents_stored": len(documents),
                "latency_ms": latency_ms,
                "embedding_tokens": _input_tokens(embedded),
                "chunks_removed": chunks_removed,
            }

        except Exception as e:
            logger.error("Document ingestion failed: %s", e)
            raise RuntimeError(f"Ingestion failed: {e}") from e

    async def search_similar(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float | None = None,
    ) -> dict:
        """
        Perform semantic similarity search on ingested documents.

        The query is embedded with ``input_type="query"``. Results are raw vector-store
        hits: access control, PII masking and reranking are applied by the API layer
        (``rag/secure_retrieval.py``), not here. Searching before anything has been
        ingested returns no results.

        Args:
            query: Query text to search for (max 100,000 characters)
            top_k: Number of results to return (1-100, default 5)
            similarity_threshold: Optional minimum similarity score (0.0-1.0)

        Returns:
            Dictionary with:
            - results: List of search results with chunk_id, text, similarity_score, metadata
            - latency_ms: Search operation time
            - embedding_tokens: Tokens used to embed the query, or None if not reported

        Raises:
            ValueError: If query is empty or too long, or top_k is invalid
            RuntimeError: If embedding or search fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if len(query) > 100000:
            raise ValueError("Text exceeds maximum length of 100,000 characters")

        if not (1 <= top_k <= 100):
            raise ValueError("top_k must be between 1 and 100")

        start_time = time.time()

        try:
            # Embed the query
            try:
                embedded = await self.embedding_provider.embed(
                    EmbeddingRequest(texts=[query], input_type=QUERY)
                )
            except Exception as e:
                raise RuntimeError(f"Embedding provider error: {e}") from e
            query_embedding = embedded.embeddings[0]

            # Search vector store
            search_results = await self.vector_store.search(
                collection=self.collection,
                vector=query_embedding.tolist()
                if isinstance(query_embedding, np.ndarray)
                else query_embedding,
                limit=top_k,
                score_threshold=similarity_threshold,
            )

            # Format results
            results = [
                {
                    "chunk_id": result.chunk_id,
                    "text": result.text,
                    "similarity_score": result.score,
                    "metadata": result.metadata,
                }
                for result in search_results
            ]

            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                "Search complete: query_len=%d, results=%d, latency_ms=%.1f",
                len(query),
                len(results),
                latency_ms,
            )

            return {
                "results": results,
                "latency_ms": latency_ms,
                "embedding_tokens": _input_tokens(embedded),
            }

        except Exception as e:
            logger.error("Search failed: %s", e)
            raise RuntimeError(f"Search failed: {e}") from e

    async def rag_query(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict:
        """
        Retrieve passages for a RAG query: embed the query and search.

        Returns passages only; generating an answer from them is the agent layer's job
        (Day 17, RFC-009 Phase 3).

        Args:
            query: User query text
            top_k: Number of results to return (1-100, default 5)

        Returns:
            Dictionary with:
            - results: List of relevant chunks (most similar first)
            - latency_ms: Total pipeline time
            - embedding_tokens: Tokens used to embed the query, or None if not reported

        Raises:
            ValueError: If query is empty or top_k is invalid
            RuntimeError: If any pipeline step fails
        """
        start_time = time.time()

        try:
            search_response = await self.search_similar(query, top_k=top_k)

            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                "RAG query complete: query_len=%d, results=%d, latency_ms=%.1f",
                len(query),
                len(search_response["results"]),
                latency_ms,
            )

            return {
                "results": search_response["results"],
                "latency_ms": latency_ms,
                "embedding_tokens": search_response.get("embedding_tokens"),
            }

        except Exception as e:
            logger.error("RAG query failed: %s", e)
            raise RuntimeError(f"RAG query failed: {e}") from e

    def _embedding_model_name(self) -> str:
        model = getattr(self.embedding_provider, "model", None)
        if not model:
            model = getattr(getattr(self.embedding_provider, "config", None), "model", None)
        return str(model or "unknown")

    async def health_check(self) -> dict:
        """
        Check health of embedding provider and vector store.

        Returns:
            Dictionary with:
            - status: "healthy" if all services operational
            - embedding_provider: Provider name
            - embedding_model: Model name (if available)
            - vector_store: Vector store type
            - collection: Collection in use
            - latency_ms: Health check duration

        Raises:
            RuntimeError: If any component is unreachable, or the collection was built
                with a different embedding model, dimension or metric
        """
        start_time = time.time()

        try:
            # Test embedding provider
            await self.embed_text("health check")
            embedding_provider_name = type(self.embedding_provider).__name__

            # Test vector store (health_check reports failure by returning False)
            if not await self.vector_store.health_check():
                raise RuntimeError("Vector store is unreachable")
            vector_store_name = type(self.vector_store).__name__

            # Fail if the collection was built with a different embedding model/dimension
            await self.vector_store.verify_collection(self.collection)

            latency_ms = (time.time() - start_time) * 1000

            logger.info("Health check passed: latency_ms=%.1f", latency_ms)

            return {
                "status": "healthy",
                "embedding_provider": embedding_provider_name,
                "embedding_model": self._embedding_model_name(),
                "vector_store": vector_store_name,
                "collection": self.collection,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            logger.error("Health check failed: %s", e)
            raise RuntimeError(f"Health check failed: {e}") from e

    async def list_collections(self) -> dict:
        """
        List all collections in the vector store.

        Returns:
            Dictionary with:
            - collections: List of collection info
                - name: Collection name
                - vector_size: Embedding dimension
                - count: Number of vectors
                - distance: Distance metric
            - latency_ms: Operation time

        Raises:
            RuntimeError: If vector store is unreachable
        """
        start_time = time.time()

        try:
            names = await self.vector_store.list_collections()
            collections = []
            for name in names:
                stats = await self.vector_store.collection_stats(name)
                collections.append(
                    {
                        "name": stats.get("name", name),
                        "vector_size": int(stats.get("vector_size") or 0),
                        "count": int(stats.get("point_count") or 0),
                        "distance": str(stats.get("distance", "unknown")),
                    }
                )

            latency_ms = (time.time() - start_time) * 1000

            logger.info("Listed %d collections", len(collections))

            return {
                "collections": collections,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            logger.error("Failed to list collections: %s", e)
            raise RuntimeError(f"Failed to list collections: {e}") from e


def _input_tokens(response: EmbeddingResponse) -> int | None:
    """Embedding tokens for a call, or None when the provider did not report usage."""
    return response.usage.input_tokens if response.usage else None
