"""RAG Service orchestration layer for embedding and vector store operations.

This module provides the RAGService class which orchestrates:
- Text embedding (via configurable EmbeddingProvider)
- Document ingestion (chunking, embedding, storage)
- Semantic search (vector similarity)
- Full RAG query pipeline

The service is provider-agnostic, supporting OpenAI, NVIDIA, and future
embedding providers through the EmbeddingProvider protocol.
"""

import logging
import time

import numpy as np

from aegis_platform.rag.chunking import create_chunker
from aegis_platform.rag.chunking.strategy import ChunkingConfig
from aegis_platform.rag.models import Document
from aegis_platform.rag.protocols import EmbeddingProvider, VectorStore

logger = logging.getLogger(__name__)


class RAGService:
    """
    Orchestration service for RAG operations.

    Coordinates embedding providers and vector stores to support:
    - Text embedding generation
    - Document ingestion and chunking
    - Semantic search
    - Full RAG query execution

    Attributes:
        embedding_provider: Configured embedding provider (OpenAI, NVIDIA, etc.)
        vector_store: Configured vector store (Qdrant, etc.)
        chunker: Text chunking strategy
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        """
        Initialize RAG Service with configured providers.

        Args:
            embedding_provider: EmbeddingProvider instance (OpenAI, NVIDIA, etc.)
            vector_store: VectorStore instance (Qdrant, etc.)

        Raises:
            ValueError: If provider configuration is invalid
        """
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

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
            "RAGService initialized with provider=%s, vector_store=%s",
            type(embedding_provider).__name__,
            type(vector_store).__name__,
        )

    async def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text input.

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
        Ingest documents: chunk, embed, and store in vector database.

        Args:
            documents: List of Document objects to ingest

        Returns:
            Dictionary with:
            - chunks_created: Number of chunks produced
            - documents_stored: Number of documents processed
            - latency_ms: Total operation time

        Raises:
            ValueError: If document list is empty
            RuntimeError: If embedding or storage fails
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
            embeddings = await self.embedding_provider.embed_batch(chunk_texts)

            logger.info("Generated embeddings for %d chunks", len(embeddings))

            # Prepare vectors for storage
            vector_ids = []
            vector_data = []
            metadata_list = []

            for chunk, embedding in zip(all_chunks, embeddings):
                vector_ids.append(chunk.chunk_id)
                vector_data.append(
                    embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                )
                metadata_list.append(
                    {
                        "text": chunk.text,
                        "doc_id": chunk.doc_id,
                        "sequence": chunk.sequence,
                        "metadata": chunk.metadata or {},
                    }
                )

            # Store vectors in vector database
            await self.vector_store.insert(
                collection="documents", vectors=vector_data, metadata=metadata_list, ids=vector_ids
            )

            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                "Ingestion complete: chunks=%d, documents=%d, latency_ms=%.1f",
                len(all_chunks),
                len(documents),
                latency_ms,
            )

            return {
                "chunks_created": len(all_chunks),
                "documents_stored": len(documents),
                "latency_ms": latency_ms,
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

        Args:
            query: Query text to search for
            top_k: Number of results to return (1-100, default 5)
            similarity_threshold: Optional minimum similarity score (0.0-1.0)

        Returns:
            Dictionary with:
            - results: List of search results with text, score, metadata
            - latency_ms: Search operation time

        Raises:
            ValueError: If query is empty or top_k is invalid
            RuntimeError: If search fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not (1 <= top_k <= 100):
            raise ValueError("top_k must be between 1 and 100")

        start_time = time.time()

        try:
            # Embed the query
            query_embedding = await self.embed_text(query)

            # Search vector store
            search_results = await self.vector_store.search(
                collection="documents",
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
        Execute full RAG query pipeline: embed query, search, return results.

        Args:
            query: User query text
            top_k: Number of results to return (1-100, default 5)

        Returns:
            Dictionary with:
            - results: List of relevant chunks (most similar first)
            - latency_ms: Total pipeline time

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
            }

        except Exception as e:
            logger.error("RAG query failed: %s", e)
            raise RuntimeError(f"RAG query failed: {e}") from e

    async def health_check(self) -> dict:
        """
        Check health of embedding provider and vector store.

        Returns:
            Dictionary with:
            - status: "healthy" if all services operational
            - embedding_provider: Provider name
            - embedding_model: Model name (if available)
            - vector_store: Vector store type
            - latency_ms: Health check duration

        Raises:
            RuntimeError: If any component is unreachable
        """
        start_time = time.time()

        try:
            # Test embedding provider
            await self.embed_text("health check")
            embedding_provider_name = type(self.embedding_provider).__name__

            # Test vector store
            await self.vector_store.health_check()
            vector_store_name = type(self.vector_store).__name__

            latency_ms = (time.time() - start_time) * 1000

            logger.info("Health check passed: latency_ms=%.1f", latency_ms)

            return {
                "status": "healthy",
                "embedding_provider": embedding_provider_name,
                "embedding_model": "unknown",
                "vector_store": vector_store_name,
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
            collections = await self.vector_store.list_collections()

            latency_ms = (time.time() - start_time) * 1000

            logger.info("Listed %d collections", len(collections))

            return {
                "collections": collections,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            logger.error("Failed to list collections: %s", e)
            raise RuntimeError(f"Failed to list collections: {e}") from e
