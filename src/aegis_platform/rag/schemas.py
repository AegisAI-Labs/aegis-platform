"""Pydantic schemas for RAG API endpoints.

Request/response models for embedding, ingestion, search, and query operations.
All models include validation rules and documentation.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from aegis_platform.rag.access_control import AccessLevel, Role


class DocumentInput(BaseModel):
    """Document for ingestion."""

    doc_id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document content")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Document metadata (source, type, etc.)"
    )
    access_level: AccessLevel = Field(
        default=AccessLevel.INTERNAL, description="Who may retrieve this document"
    )
    allowed_roles: list[Role] = Field(
        default_factory=list, description="Roles allowed to read RESTRICTED documents"
    )
    organization_id: str | None = Field(
        default=None,
        description="Owning tenant; defaults to the caller's X-Organization-Id header",
    )

    @field_validator("doc_id")
    @classmethod
    def validate_doc_id(cls, v: str) -> str:
        """Validate doc_id is not empty."""
        if not v or not v.strip():
            raise ValueError("doc_id cannot be empty")
        return v.strip()

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content is not empty."""
        if not v or not v.strip():
            raise ValueError("content cannot be empty")
        if len(v) > 1_000_000:
            raise ValueError("content exceeds maximum length of 1,000,000 characters")
        return v.strip()


class EmbedRequest(BaseModel):
    """Request to embed text."""

    text: str = Field(..., description="Text to embed", min_length=1, max_length=100_000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate text is not empty."""
        if not v or not v.strip():
            raise ValueError("text cannot be empty")
        return v.strip()


class EmbedResponse(BaseModel):
    """Response with text embedding."""

    embedding: list[float] = Field(..., description="Vector embedding")
    dimension: int = Field(..., description="Embedding dimension")
    latency_ms: float = Field(..., description="Embedding latency in milliseconds", ge=0)


class IngestRequest(BaseModel):
    """Request to ingest documents."""

    documents: list[DocumentInput] = Field(
        ...,
        description="Documents to ingest",
        min_length=1,
        max_length=100,
    )


class IngestResponse(BaseModel):
    """Response after document ingestion."""

    chunks_created: int = Field(..., description="Total chunks created", ge=0)
    documents_stored: int = Field(..., description="Documents processed", ge=0)
    latency_ms: float = Field(..., description="Ingestion latency in milliseconds", ge=0)
    chunks_removed: int = Field(
        default=0,
        description="Chunks from previous versions of these documents that were removed",
        ge=0,
    )


class SearchResult(BaseModel):
    """Single search result."""

    chunk_id: str = Field(..., description="Chunk identifier")
    text: str = Field(..., description="Chunk text content")
    similarity_score: float = Field(
        ...,
        description="Similarity score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata",
    )
    rerank_score: float | None = Field(
        default=None,
        description="Cross-encoder relevance (0.0-1.0) when rerank=true",
        ge=0.0,
        le=1.0,
    )


class SearchRequest(BaseModel):
    """Request to search vector store."""

    query: str = Field(..., description="Query text", min_length=1, max_length=10_000)
    top_k: int = Field(
        default=5,
        description="Number of results to return",
        ge=1,
        le=100,
    )
    similarity_threshold: float | None = Field(
        default=None,
        description="Minimum similarity score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty."""
        if not v or not v.strip():
            raise ValueError("query cannot be empty")
        return v.strip()


class SearchResponse(BaseModel):
    """Response with search results."""

    results: list[SearchResult] = Field(
        default_factory=list,
        description="Search results ranked by similarity",
    )
    latency_ms: float = Field(
        ...,
        description="Search latency in milliseconds",
        ge=0,
    )


class QueryRequest(BaseModel):
    """Request for full RAG query."""

    query: str = Field(..., description="Query text", min_length=1, max_length=10_000)
    top_k: int = Field(
        default=5,
        description="Number of results to return",
        ge=1,
        le=100,
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty."""
        if not v or not v.strip():
            raise ValueError("query cannot be empty")
        return v.strip()


class QueryResponse(BaseModel):
    """Response with RAG query results."""

    results: list[SearchResult] = Field(
        default_factory=list,
        description="Query results with context passages",
    )
    latency_ms: float = Field(
        ...,
        description="Query latency in milliseconds",
        ge=0,
    )


class CollectionInfo(BaseModel):
    """Information about a vector store collection."""

    name: str = Field(..., description="Collection name")
    vector_size: int = Field(..., description="Embedding dimension", ge=1)
    count: int = Field(..., description="Number of vectors", ge=0)
    distance: str = Field(..., description="Distance metric (cosine, euclidean, etc.)")


class CollectionsResponse(BaseModel):
    """Response with list of collections."""

    collections: list[CollectionInfo] = Field(
        default_factory=list,
        description="Available collections",
    )


class HealthCheckResponse(BaseModel):
    """Response from health check endpoint."""

    status: str = Field(
        ...,
        description='Service status ("healthy" or error message)',
    )
    embedding_provider: str = Field(..., description="Embedding provider name")
    embedding_model: str = Field(..., description="Embedding model name")
    vector_store: str = Field(..., description="Vector store type")
    collection: str | None = Field(default=None, description="Collection in use")
    latency_ms: float = Field(..., description="Health check latency", ge=0)


class ErrorDetail(BaseModel):
    """Error response detail."""

    detail: str = Field(..., description="Error message")
    error_code: str | None = Field(None, description="Error code for programmatic handling")


class VerifySource(BaseModel):
    """A passage that was given to the LLM as context."""

    doc_id: str = Field(..., min_length=1, description="Source document identifier")
    text: str = Field(..., min_length=1, max_length=100_000, description="Passage text")
    title: str | None = Field(default=None, description="Document title (defaults to doc_id)")
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Retrieval score")


class VerifyRequest(BaseModel):
    """Request to check an LLM answer against its sources."""

    query: str = Field(..., min_length=1, max_length=10_000, description="User query")
    answer: str = Field(..., min_length=1, max_length=100_000, description="LLM answer")
    sources: list[VerifySource] = Field(
        default_factory=list, max_length=100, description="Passages provided to the LLM"
    )


class CitationOut(BaseModel):
    """One sentence of the answer and the sources that support it."""

    claim: str
    doc_ids: list[str]
    confidence: float


class VerifyResponse(BaseModel):
    """Citation and hallucination assessment of an answer."""

    citations: list[CitationOut] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall grounding confidence")
    risk_level: str = Field(..., description='Hallucination risk: "low", "medium" or "high"')
    risk_score: float = Field(..., ge=0.0, le=1.0)
    patterns_detected: list[str] = Field(default_factory=list)
    explanation: str
    latency_ms: float = Field(..., ge=0)
