"""Data models for RAG (Retrieval-Augmented Generation) system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class EmbeddingRequest:
    """Request for embedding generation."""

    text: str
    model: str | None = None


@dataclass
class EmbeddingResponse:
    """Response containing generated embeddings."""

    text: str
    embedding: list[float]
    model: str
    dimension: int
    tokens_used: int = 0


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    chunk_id: str
    doc_id: str
    sequence: int
    start_pos: int
    end_pos: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """A document with metadata and chunks."""

    id: str
    title: str
    content: str
    source: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    chunks: list[Chunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result from vector store search."""

    document_id: str
    chunk_id: str
    text: str
    score: float
    distance: float
    metadata: dict[str, Any] = field(default_factory=dict)
