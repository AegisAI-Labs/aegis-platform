"""Data models for RAG (Retrieval-Augmented Generation) system."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np

QUERY: Literal["query"] = "query"
PASSAGE: Literal["passage"] = "passage"
InputType = Literal["query", "passage"]


@dataclass(frozen=True)
class EmbeddingRequest:
    """Texts to embed in one logical call (ADR-013).

    ``input_type`` tells asymmetric retrieval models whether the texts are search queries or
    stored passages; symmetric providers ignore it.
    """

    texts: list[str]
    input_type: InputType = PASSAGE


@dataclass(frozen=True)
class EmbeddingUsage:
    """Token usage for a whole request (summed over the API calls it was split into).

    ``None`` means the provider did not report usage for at least one call.
    """

    input_tokens: int | None = None


@dataclass(frozen=True)
class EmbeddingResponse:
    """Vectors aligned with ``EmbeddingRequest.texts`` plus serving metadata."""

    embeddings: list[np.ndarray]
    model: str
    provider: str
    dimension: int
    usage: EmbeddingUsage | None = None


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
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
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
