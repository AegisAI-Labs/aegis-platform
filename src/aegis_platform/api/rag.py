"""RAG REST API endpoints for embedding, ingestion, search, and query operations.

Provides 6 FastAPI endpoints:
1. POST /api/rag/health - Service health check
2. POST /api/rag/embed - Generate embeddings for text
3. POST /api/rag/ingest - Ingest documents (chunk, embed, store)
4. POST /api/rag/search - Semantic similarity search
5. POST /api/rag/query - Full RAG pipeline (embed query + search)
6. GET /api/rag/collections - List vector store collections

Configuration-driven provider selection (NVIDIA/OpenAI via RAG_EMBEDDING_PROVIDER env var).
"""

import logging
import os
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from aegis_platform.rag.models import Document
from aegis_platform.rag.protocols import EmbeddingProvider, VectorStore
from aegis_platform.rag.schemas import (
    CollectionsResponse,
    EmbedRequest,
    EmbedResponse,
    ErrorDetail,
    HealthCheckResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
)
from aegis_platform.rag.service import RAGService

logger = logging.getLogger(__name__)

# Create FastAPI router
router = APIRouter()


def get_configured_embedding_provider() -> EmbeddingProvider:
    """
    Load embedding provider from configuration.

    Uses RAG_EMBEDDING_PROVIDER env var to select provider.
    Supported: "openai", "nvidia"

    Returns:
        Configured EmbeddingProvider instance

    Raises:
        ValueError: If provider is unknown or configuration invalid
    """
    from aegis_platform.rag.embeddings import OpenAIEmbeddingConfig
    from aegis_platform.rag.provider_factory import (
        create_embedding_provider,
    )

    provider_name = os.getenv("RAG_EMBEDDING_PROVIDER", "openai").lower()

    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        config = OpenAIEmbeddingConfig(api_key=api_key)
        return create_embedding_provider(config)
    else:
        raise ValueError(f"Unknown embedding provider: {provider_name}")


def get_configured_vector_store() -> VectorStore:
    """
    Load vector store from configuration.

    Uses RAG_VECTOR_STORE_* env vars for connection details.
    Currently supports: "qdrant"

    Returns:
        Configured VectorStore instance

    Raises:
        ValueError: If vector store is unknown or configuration invalid
    """
    from aegis_platform.rag.provider_factory import (
        create_vector_store,
    )
    from aegis_platform.rag.storage import QdrantVectorStoreConfig

    store_type = os.getenv("RAG_VECTOR_STORE", "qdrant").lower()

    if store_type == "qdrant":
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")
        config = QdrantVectorStoreConfig(url=url, api_key=api_key)
        return create_vector_store(config)
    else:
        raise ValueError(f"Unknown vector store: {store_type}")


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """
    Singleton RAG Service with configured providers.

    Lazy initialization on first use. Cached for reuse across requests.
    Configuration sourced from environment variables.

    Returns:
        RAGService with configured embedding provider and vector store

    Raises:
        ValueError: If provider configuration is invalid
        RuntimeError: If providers are unreachable
    """
    try:
        embedding_provider = get_configured_embedding_provider()
        vector_store = get_configured_vector_store()

        logger.info(
            "Initializing RAG Service with provider=%s, vector_store=%s",
            type(embedding_provider).__name__,
            type(vector_store).__name__,
        )

        return RAGService(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )
    except Exception as e:
        logger.error("Failed to initialize RAG Service: %s", e)
        raise RuntimeError(f"RAG Service initialization failed: {e}") from e


# ============================================================================
# ENDPOINT 1: Health Check
# ============================================================================


@router.post(
    "/health",
    response_model=HealthCheckResponse,
    summary="RAG Service Health Check",
    tags=["rag"],
    responses={
        200: {"description": "Service is healthy"},
        503: {"model": ErrorDetail, "description": "Service unavailable"},
    },
)
async def health_check(service: RAGService = Depends(get_rag_service)) -> HealthCheckResponse:
    """
    Check health of RAG Service and its dependencies.

    Verifies:
    - Embedding provider connectivity
    - Vector store connectivity

    Returns provider and vector store information if healthy.

    Raises:
        HTTPException (503): If any dependency is unreachable
    """
    try:
        result = await service.health_check()
        return HealthCheckResponse(**result)
    except RuntimeError as e:
        logger.error("Health check failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Unexpected error in health check: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Health check failed",
        ) from e


# ============================================================================
# ENDPOINT 2: Embed Text
# ============================================================================


@router.post(
    "/embed",
    response_model=EmbedResponse,
    summary="Generate Text Embeddings",
    tags=["rag"],
    responses={
        200: {"description": "Embedding generated successfully"},
        400: {"model": ErrorDetail, "description": "Invalid request (empty text, etc.)"},
        503: {"model": ErrorDetail, "description": "Embedding provider unavailable"},
    },
)
async def embed_text(
    request: EmbedRequest,
    service: RAGService = Depends(get_rag_service),
) -> EmbedResponse:
    """
    Generate embeddings for input text.

    Args:
        request: EmbedRequest with text to embed

    Returns:
        EmbedResponse with embedding vector, dimension, and latency

    Raises:
        HTTPException (400): If text is empty or too long
        HTTPException (503): If embedding provider is unavailable
    """
    try:
        embedding = await service.embed_text(request.text)

        # Convert numpy array to list if needed
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

        logger.info(
            "Successfully embedded text: len=%d, dim=%d",
            len(request.text),
            len(embedding_list),
        )

        return EmbedResponse(
            embedding=embedding_list,
            dimension=len(embedding_list),
            latency_ms=0,  # Set by service
        )
    except ValueError as e:
        logger.warning("Embedding request validation failed: %s", e)
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e
    except RuntimeError as e:
        logger.error("Embedding failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Unexpected error in embed endpoint: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Embedding failed",
        ) from e


# ============================================================================
# ENDPOINT 3: Ingest Documents
# ============================================================================


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest Documents",
    tags=["rag"],
    responses={
        200: {"description": "Documents ingested successfully"},
        400: {"model": ErrorDetail, "description": "Invalid request"},
        503: {"model": ErrorDetail, "description": "Service unavailable"},
    },
)
async def ingest_documents(
    request: IngestRequest,
    service: RAGService = Depends(get_rag_service),
) -> IngestResponse:
    """
    Ingest documents: chunk, embed, and store in vector database.

    Pipeline:
    1. Chunk documents using configured strategy
    2. Generate embeddings for all chunks
    3. Store chunks and embeddings in vector database

    Args:
        request: IngestRequest with list of documents
        service: RAGService instance (injected)

    Returns:
        IngestResponse with chunk and document counts, latency

    Raises:
        HTTPException (400): If document list is empty or invalid
        HTTPException (503): If ingestion fails
    """
    try:
        # Convert DocumentInput objects to Document domain model
        documents = [
            Document(
                id=doc.doc_id,
                title=doc.doc_id,  # Use doc_id as title
                content=doc.content,
                source="api",
                metadata=doc.metadata or {},
            )
            for doc in request.documents
        ]

        result = await service.ingest_documents(documents)

        logger.info(
            "Documents ingested: docs=%d, chunks=%d, latency_ms=%.1f",
            result["documents_stored"],
            result["chunks_created"],
            result["latency_ms"],
        )

        return IngestResponse(**result)
    except ValueError as e:
        logger.warning("Ingest request validation failed: %s", e)
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e
    except RuntimeError as e:
        logger.error("Ingestion failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Unexpected error in ingest endpoint: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Ingestion failed",
        ) from e


# ============================================================================
# ENDPOINT 4: Vector Search
# ============================================================================


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic Similarity Search",
    tags=["rag"],
    responses={
        200: {"description": "Search completed successfully"},
        400: {"model": ErrorDetail, "description": "Invalid request"},
        503: {"model": ErrorDetail, "description": "Search failed"},
    },
)
async def search_vectors(
    request: SearchRequest,
    service: RAGService = Depends(get_rag_service),
) -> SearchResponse:
    """
    Perform semantic similarity search on ingested documents.

    Pipeline:
    1. Embed the query text
    2. Search vector store for similar chunks
    3. Return top-k results sorted by similarity

    Args:
        request: SearchRequest with query and optional top_k
        service: RAGService instance (injected)

    Returns:
        SearchResponse with search results and latency

    Raises:
        HTTPException (400): If query is empty or top_k invalid
        HTTPException (503): If search fails
    """
    try:
        result = await service.search_similar(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
        )

        logger.info(
            "Search completed: query_len=%d, results=%d, latency_ms=%.1f",
            len(request.query),
            len(result["results"]),
            result["latency_ms"],
        )

        return SearchResponse(**result)
    except ValueError as e:
        logger.warning("Search request validation failed: %s", e)
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e
    except RuntimeError as e:
        logger.error("Search failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Unexpected error in search endpoint: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Search failed",
        ) from e


# ============================================================================
# ENDPOINT 5: Full RAG Query
# ============================================================================


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Full RAG Query Pipeline",
    tags=["rag"],
    responses={
        200: {"description": "Query executed successfully"},
        400: {"model": ErrorDetail, "description": "Invalid request"},
        503: {"model": ErrorDetail, "description": "Query failed"},
    },
)
async def rag_query(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """
    Execute full RAG query pipeline.

    Pipeline:
    1. Embed the user query
    2. Search vector store for similar chunks
    3. Return relevant chunks with metadata for LLM context

    This is the primary RAG endpoint used for context retrieval.

    Args:
        request: QueryRequest with query text and optional top_k
        service: RAGService instance (injected)

    Returns:
        QueryResponse with search results and latency

    Raises:
        HTTPException (400): If query is empty or top_k invalid
        HTTPException (503): If query execution fails
    """
    try:
        result = await service.rag_query(
            query=request.query,
            top_k=request.top_k,
        )

        logger.info(
            "RAG query completed: query_len=%d, results=%d, latency_ms=%.1f",
            len(request.query),
            len(result["results"]),
            result["latency_ms"],
        )

        return QueryResponse(**result)
    except ValueError as e:
        logger.warning("Query request validation failed: %s", e)
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e
    except RuntimeError as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Unexpected error in query endpoint: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Query failed",
        ) from e


# ============================================================================
# ENDPOINT 6: List Collections
# ============================================================================


@router.get(
    "/collections",
    response_model=CollectionsResponse,
    summary="List Vector Store Collections",
    tags=["rag"],
    responses={
        200: {"description": "Collections listed successfully"},
        503: {"model": ErrorDetail, "description": "Service unavailable"},
    },
)
async def list_collections(
    service: RAGService = Depends(get_rag_service),
) -> CollectionsResponse:
    """
    List all collections in the vector store.

    Returns metadata about each collection including:
    - Collection name
    - Vector dimension
    - Number of vectors stored
    - Distance metric

    Args:
        service: RAGService instance (injected)

    Returns:
        CollectionsResponse with list of collections

    Raises:
        HTTPException (503): If unable to query vector store
    """
    try:
        result = await service.list_collections()

        logger.info(
            "Listed collections: count=%d, latency_ms=%.1f",
            len(result["collections"]),
            result["latency_ms"],
        )

        return CollectionsResponse(**result)
    except RuntimeError as e:
        logger.error("Failed to list collections: %s", e)
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Unexpected error in collections endpoint: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Failed to list collections",
        ) from e


# ============================================================================
# Router Summary
# ============================================================================

"""
Router Summary:
- POST /api/rag/health - Service health check
- POST /api/rag/embed - Generate embeddings
- POST /api/rag/ingest - Ingest documents
- POST /api/rag/search - Semantic search
- POST /api/rag/query - Full RAG pipeline
- GET /api/rag/collections - List collections

All endpoints use dependency injection with @lru_cache singleton pattern
for RAGService initialization (ADR-010).

Configuration:
- RAG_EMBEDDING_PROVIDER: "openai" or "nvidia" (default: "openai")
- RAG_EMBEDDING_MODEL: Model name (e.g., "gpt-4", "nvidia/nemotron-3.5-embed-1b")
- RAG_EMBEDDING_KEY: API key for embedding provider
- RAG_VECTOR_STORE_URL: Qdrant endpoint
- RAG_VECTOR_STORE_API_KEY: Qdrant API key

Error Handling:
- 400: Invalid request (validation failure, empty input, etc.)
- 401: Authentication failure (invalid credentials)
- 503: Service unavailable (provider unreachable)

All responses include latency_ms for observability.
"""
