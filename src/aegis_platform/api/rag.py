"""RAG REST API endpoints for embedding, ingestion, search, query and verification.

Provides 7 FastAPI endpoints:
1. POST /rag/health - Service health check
2. POST /rag/embed - Generate embeddings for text
3. POST /rag/ingest - Ingest documents with access metadata (chunk, embed, store)
4. POST /rag/search - Semantic search through the safety pipeline
5. POST /rag/query - Retrieval step of RAG through the safety pipeline
6. GET /rag/collections - List vector store collections
7. POST /rag/verify - Citation and hallucination check of an LLM answer

Search and query apply guardrails, access control and PII masking on every call;
query enhancement (?enhance=true) and cross-encoder reranking (?rerank=true) are opt-in.

Caller identity comes from trusted X-User-Id / X-User-Role / X-Organization-Id headers,
which an authenticating gateway must set. Requests without them are treated as GUEST.
"""

import logging
import os
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, Depends, Header, HTTPException, Query

from aegis_platform.rag.access_control import Role, UserContext
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
    VerifyRequest,
    VerifyResponse,
)
from aegis_platform.rag.secure_retrieval import (
    QueryRejectedError,
    access_metadata,
    secure_retrieve,
    verify_answer,
)
from aegis_platform.rag.service import DEFAULT_COLLECTION, RAGService

logger = logging.getLogger(__name__)

# Create FastAPI router
router = APIRouter()


def _env(*names: str) -> str | None:
    """Return the first non-empty environment variable among ``names``."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def get_configured_embedding_provider() -> EmbeddingProvider:
    """
    Load embedding provider from configuration.

    Uses RAG_EMBEDDING_PROVIDER env var to select provider:
    - "openai": key from RAG_EMBEDDING_KEY, falling back to OPENAI_API_KEY
    - "nvidia": all settings from RAG_EMBEDDING_* (see NVIDIAEmbeddingConfig.from_env)

    Returns:
        Configured EmbeddingProvider instance

    Raises:
        ValueError: If provider is unknown or configuration invalid
    """
    from aegis_platform.rag.embeddings import NVIDIAEmbeddingConfig, OpenAIEmbeddingConfig
    from aegis_platform.rag.provider_factory import (
        create_embedding_provider,
    )

    provider_name = (_env("RAG_EMBEDDING_PROVIDER") or "openai").lower()

    if provider_name == "openai":
        api_key = _env("RAG_EMBEDDING_KEY", "OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set RAG_EMBEDDING_KEY or OPENAI_API_KEY")
        return create_embedding_provider(OpenAIEmbeddingConfig(api_key=api_key))
    if provider_name == "nvidia":
        return create_embedding_provider(NVIDIAEmbeddingConfig.from_env())
    raise ValueError(f"Unknown embedding provider: {provider_name}")


def embedding_dimension_for(provider: EmbeddingProvider) -> int:
    """Vector size produced by ``provider``; the vector store must use the same size."""
    config = getattr(provider, "config", None)
    for attribute in ("dimension", "embedding_dimension"):
        value = getattr(config, attribute, None)
        if isinstance(value, int) and value > 0:
            return value
    return int(_env("RAG_EMBEDDING_DIMENSION") or 1536)


def embedding_model_for(provider: EmbeddingProvider) -> str | None:
    """Model name of ``provider``; recorded on new collections and checked on existing ones."""
    model = getattr(provider, "model", None) or getattr(
        getattr(provider, "config", None), "model", None
    )
    return str(model) if model else None


def get_configured_collection() -> str:
    """Collection for RAG data (RAG_COLLECTION, default "documents").

    Use a separate collection per embedding model; switching models means re-ingesting
    into a new collection.
    """
    return _env("RAG_COLLECTION") or DEFAULT_COLLECTION


def get_configured_vector_store(
    embedding_dimension: int | None = None,
    embedding_model: str | None = None,
) -> VectorStore:
    """
    Load vector store from configuration.

    Reads RAG_VECTOR_STORE_URL / RAG_VECTOR_STORE_API_KEY (falling back to QDRANT_URL /
    QDRANT_API_KEY) and RAG_VECTOR_STORE_DISTANCE_METRIC. ``embedding_dimension`` must match
    the embedding provider (falls back to RAG_EMBEDDING_DIMENSION, then 1536).
    Use ``RAG_VECTOR_STORE_URL=:memory:`` for an in-process store (local experiments).
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

    store_type = (_env("RAG_VECTOR_STORE") or "qdrant").lower()

    if store_type == "qdrant":
        config = QdrantVectorStoreConfig(
            url=_env("RAG_VECTOR_STORE_URL", "QDRANT_URL") or "http://localhost:6333",
            api_key=_env("RAG_VECTOR_STORE_API_KEY", "QDRANT_API_KEY"),
            distance_metric=_env("RAG_VECTOR_STORE_DISTANCE_METRIC") or "cosine",
            embedding_dimension=embedding_dimension or int(_env("RAG_EMBEDDING_DIMENSION") or 1536),
            embedding_model=embedding_model,
        )
        return create_vector_store(config)
    else:
        raise ValueError(f"Unknown vector store: {store_type}")


def get_user_context(
    x_user_id: str | None = Header(default=None),
    x_user_role: str | None = Header(default=None),
    x_organization_id: str | None = Header(default=None),
) -> UserContext:
    """
    Build the caller's identity from trusted gateway headers.

    Requests without X-User-Id and X-User-Role are treated as an anonymous GUEST,
    which can only read PUBLIC documents.

    Raises:
        HTTPException (400): If X-User-Role is not a known role
    """
    if not x_user_id or not x_user_role:
        return UserContext(user_id="anonymous", role=Role.GUEST)

    try:
        role = Role(x_user_role.strip().lower())
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown X-User-Role {x_user_role!r}; expected one of "
            f"{[r.value for r in Role]}",
        ) from e

    return UserContext(user_id=x_user_id, role=role, organization_id=x_organization_id or None)


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
        vector_store = get_configured_vector_store(
            embedding_dimension_for(embedding_provider),
            embedding_model_for(embedding_provider),
        )
        collection = get_configured_collection()

        logger.info(
            "Initializing RAG Service with provider=%s, vector_store=%s",
            type(embedding_provider).__name__,
            type(vector_store).__name__,
        )

        return RAGService(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            collection=collection,
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
        403: {"model": ErrorDetail, "description": "Caller may not ingest these documents"},
        503: {"model": ErrorDetail, "description": "Service unavailable"},
    },
)
async def ingest_documents(
    request: IngestRequest,
    service: RAGService = Depends(get_rag_service),
    user: UserContext = Depends(get_user_context),
) -> IngestResponse:
    """
    Ingest documents: chunk, embed, and store in vector database.

    Pipeline (see RAGService.ingest_documents):
    1. Chunk documents using configured strategy
    2. Generate embeddings for all chunks (as passages)
    3. Upsert chunks, embeddings and access metadata into the vector database
    4. Remove chunks left from a previous version of each document

    Re-ingesting a doc_id replaces that document; ``chunks_removed`` in the response
    reports how many outdated chunks were deleted.

    Each document's access_level, allowed_roles and organization_id are stored with
    its chunks and enforced by /search and /query. organization_id defaults to the
    caller's X-Organization-Id.

    Args:
        request: IngestRequest with list of documents
        service: RAGService instance (injected)
        user: Caller identity (injected from headers)

    Returns:
        IngestResponse with chunks created, documents stored, chunks removed, latency

    Raises:
        HTTPException (400): If document list is empty or invalid
        HTTPException (403): If the caller is a guest or targets another organization
        HTTPException (503): If ingestion fails
    """
    if user.role == Role.GUEST:
        raise HTTPException(status_code=403, detail="Guests cannot ingest documents")

    documents = []
    for doc in request.documents:
        organization_id = doc.organization_id or user.organization_id
        if (
            user.role != Role.ADMIN
            and organization_id
            and user.organization_id
            and organization_id != user.organization_id
        ):
            raise HTTPException(
                status_code=403,
                detail=f"Cannot ingest {doc.doc_id!r} into another organization",
            )
        documents.append(
            Document(
                id=doc.doc_id,
                title=doc.doc_id,  # Use doc_id as title
                content=doc.content,
                source="api",
                metadata={
                    **(doc.metadata or {}),
                    **access_metadata(doc.access_level, doc.allowed_roles, organization_id),
                },
            )
        )

    try:
        result = await service.ingest_documents(documents)

        logger.info(
            "Documents ingested: docs=%d, chunks=%d, removed=%d, latency_ms=%.1f",
            result["documents_stored"],
            result["chunks_created"],
            result.get("chunks_removed", 0),
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
        400: {"model": ErrorDetail, "description": "Invalid request or rejected by guardrails"},
        503: {"model": ErrorDetail, "description": "Search failed"},
    },
)
async def search_vectors(
    request: SearchRequest,
    rerank: bool = Query(default=False, description="Rerank hits with a cross-encoder"),
    enhance: bool = Query(default=False, description="Also search LLM query expansions"),
    service: RAGService = Depends(get_rag_service),
    user: UserContext = Depends(get_user_context),
) -> SearchResponse:
    """
    Perform semantic similarity search through the safety pipeline.

    Pipeline:
    1. Guardrails reject injection attempts (400)
    2. Optional LLM query expansion (?enhance=true); expansions are re-validated
    3. Vector search
    4. Access control drops hits the caller may not read
    5. Optional cross-encoder reranking (?rerank=true)
    6. PII in returned passages is masked

    Args:
        request: SearchRequest with query, top_k and optional similarity_threshold
        rerank: Rerank with the cross-encoder
        enhance: Expand the query with the LLM
        service: RAGService instance (injected)
        user: Caller identity (injected from headers)

    Returns:
        SearchResponse with search results and latency

    Raises:
        HTTPException (400): If query is invalid or rejected by guardrails
        HTTPException (503): If search, reranking or PII masking fails
    """
    try:
        result = await secure_retrieve(
            service,
            query=request.query,
            user=user,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            rerank=rerank,
            enhance=enhance,
        )

        logger.info(
            "Search completed: query_len=%d, results=%d, latency_ms=%.1f",
            len(request.query),
            len(result["results"]),
            result["latency_ms"],
        )

        return SearchResponse(**result)
    except QueryRejectedError as e:
        raise _rejected(e) from e
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
        400: {"model": ErrorDetail, "description": "Invalid request or rejected by guardrails"},
        503: {"model": ErrorDetail, "description": "Query failed"},
    },
)
async def rag_query(
    request: QueryRequest,
    rerank: bool = Query(default=False, description="Rerank hits with a cross-encoder"),
    enhance: bool = Query(default=False, description="Also search LLM query expansions"),
    service: RAGService = Depends(get_rag_service),
    user: UserContext = Depends(get_user_context),
) -> QueryResponse:
    """
    Execute the retrieval step of RAG through the safety pipeline.

    Same pipeline as /search (guardrails, optional enhancement, access control,
    optional reranking, PII masking) without a similarity threshold. Returns the
    context passages for the LLM; answer generation happens in the agent layer.

    Args:
        request: QueryRequest with query text and optional top_k
        rerank: Rerank with the cross-encoder
        enhance: Expand the query with the LLM
        service: RAGService instance (injected)
        user: Caller identity (injected from headers)

    Returns:
        QueryResponse with search results and latency

    Raises:
        HTTPException (400): If query is invalid or rejected by guardrails
        HTTPException (503): If query execution fails
    """
    try:
        result = await secure_retrieve(
            service,
            query=request.query,
            user=user,
            top_k=request.top_k,
            rerank=rerank,
            enhance=enhance,
        )

        logger.info(
            "RAG query completed: query_len=%d, results=%d, latency_ms=%.1f",
            len(request.query),
            len(result["results"]),
            result["latency_ms"],
        )

        return QueryResponse(**result)
    except QueryRejectedError as e:
        raise _rejected(e) from e
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
# ENDPOINT 7: Verify Answer
# ============================================================================


@router.post(
    "/verify",
    response_model=VerifyResponse,
    summary="Citation and Hallucination Check",
    tags=["rag"],
    responses={
        200: {"description": "Answer assessed"},
        400: {"model": ErrorDetail, "description": "Invalid request"},
    },
)
async def verify(request: VerifyRequest) -> VerifyResponse:
    """
    Check an LLM answer against the passages it was given.

    Maps each sentence of the answer to supporting sources, computes an overall
    grounding confidence, and scores hallucination risk. With no sources the
    answer is reported as high risk. Runs offline; no provider call is made.

    Raises:
        HTTPException (400): If the answer is blank
    """
    try:
        result = await verify_answer(
            request.query,
            request.answer,
            [(s.doc_id, s.title or s.doc_id, s.text, s.score) for s in request.sources],
        )
        return VerifyResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _rejected(error: QueryRejectedError) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "message": str(error),
            "risk_level": error.risk_level,
            "violations": error.violations,
        },
    )


# ============================================================================
# Router Summary
# ============================================================================

"""
Router Summary (mounted at /rag):
- POST /rag/health - Service health check
- POST /rag/embed - Generate embeddings
- POST /rag/ingest - Ingest documents with access metadata
- POST /rag/search - Safe semantic search (?rerank=true, ?enhance=true)
- POST /rag/query - Safe retrieval for RAG (?rerank=true, ?enhance=true)
- GET /rag/collections - List collections
- POST /rag/verify - Citation and hallucination check

All endpoints use dependency injection with @lru_cache singleton pattern
for RAGService initialization (ADR-010).

Configuration:
- RAG_EMBEDDING_PROVIDER: "openai" (default) or "nvidia"
- RAG_EMBEDDING_KEY or OPENAI_API_KEY: API key for the embedding provider
- RAG_VECTOR_STORE_URL or QDRANT_URL: Qdrant endpoint
- RAG_VECTOR_STORE_API_KEY or QDRANT_API_KEY: Qdrant API key
- RAG_VECTOR_STORE_DISTANCE_METRIC: cosine (default), euclidean, manhattan
- RAG_COLLECTION: collection name (default "documents"); use one per embedding model

Identity headers: X-User-Id, X-User-Role, X-Organization-Id (trusted; set by gateway).

Error Handling:
- 400: Invalid request or query rejected by guardrails
- 403: Ingestion not permitted for the caller
- 422: Request schema validation failure
- 503: Service unavailable (provider unreachable, PII masking unavailable)

All responses include latency_ms for observability.
"""
