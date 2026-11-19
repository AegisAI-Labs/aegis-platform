"""Base reranking interface for RAG result optimization.

This module provides abstract interfaces and common types for reranking
implementations that optimize retrieval results based on query-document
relevance.

Features:
- Abstract reranker base class
- Pluggable reranking strategies (dense, sparse, hybrid)
- Configuration management
- Type-safe result models
- Async support

Example:
    >>> from aegis_platform.rag.reranker_base import (
    ...     RankerBase, RankedResult, RerankerConfig
    ... )
    >>>
    >>> # Implement custom reranker
    >>> class MyReranker(RankerBase):
    ...     async def rerank(self, query: str, results, top_k: int):
    ...         # Custom reranking logic
    ...         pass
    ...
    ...     async def score(self, query: str, document: str) -> float:
    ...         # Custom scoring logic
    ...         pass
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class RankerStrategy(str, Enum):
    """Available reranking strategies."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class RankedResult(BaseModel):
    """Result after reranking with relevance score."""

    result_id: str = Field(..., description="Unique result identifier")
    text: str = Field(..., description="Result text content")
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized relevance score (0-1)"
    )
    source_id: str | None = Field(None, description="Source document identifier")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")
    original_rank: int = Field(..., description="Original rank before reranking")
    new_rank: int = Field(..., description="New rank after reranking")


class RerankerConfig(BaseModel):
    """Configuration for reranking operations."""

    model_config = ConfigDict(validate_assignment=True)

    strategy: RankerStrategy = Field(
        default=RankerStrategy.DENSE, description="Reranking strategy to use"
    )
    batch_size: int = Field(default=32, ge=1, description="Batch size for inference")
    score_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum score to include result"
    )
    top_k: int = Field(default=10, ge=1, description="Number of top results to return")
    normalize_scores: bool = Field(default=True, description="Normalize scores to 0-1 range")
    return_metadata: bool = Field(default=True, description="Include metadata in results")


class RawResult(BaseModel):
    """Raw search result before reranking."""

    result_id: str = Field(..., description="Unique result identifier")
    text: str = Field(..., description="Result text content")
    similarity_score: float | None = Field(None, description="Vector similarity score from search")
    source_id: str | None = Field(None, description="Source document ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Result metadata")


class RankerBase(ABC):
    """Abstract base class for all reranking implementations.

    Defines the interface that all reranker implementations must follow.
    Rerankers optimize search results by computing query-document relevance.
    """

    def __init__(self, config: RerankerConfig | None = None) -> None:
        """Initialize reranker with configuration.

        Args:
            config: Reranker configuration (uses defaults if None)
        """
        self.config = config or RerankerConfig()
        logger.debug(
            f"Reranker initialized with strategy={self.config.strategy}, "
            f"batch_size={self.config.batch_size}"
        )

    @abstractmethod
    async def rerank(
        self, query: str, results: list[RawResult], top_k: int | None = None
    ) -> list[RankedResult]:
        """Rerank search results based on query relevance.

        Args:
            query: Search query string
            results: Raw search results to rerank
            top_k: Number of results to return (uses config default if None)

        Returns:
            List of reranked results sorted by relevance score (descending)

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If reranking fails
        """

    @abstractmethod
    async def score(self, query: str, document: str) -> float:
        """Compute relevance score for query-document pair.

        Args:
            query: Query string
            document: Document string

        Returns:
            Relevance score (0-1 range if normalize_scores=True)

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If scoring fails
        """

    async def batch_score(self, query: str, documents: list[str]) -> list[float]:
        """Score multiple documents for a single query.

        Default implementation calls score() sequentially. Override for
        vectorized implementations.

        Args:
            query: Query string
            documents: List of document strings

        Returns:
            List of relevance scores (one per document)

        Raises:
            ValueError: If inputs are invalid
        """
        if not documents:
            return []

        scores = []
        for doc in documents:
            score = await self.score(query, doc)
            scores.append(score)

        return scores

    def _validate_inputs(self, query: str, results: list[RawResult]) -> None:
        """Validate query and results for reranking.

        Args:
            query: Query string to validate
            results: Results list to validate

        Raises:
            TypeError: If types are invalid
            ValueError: If inputs are invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not results:
            raise ValueError("Results list cannot be empty")

        if not isinstance(results, list):
            raise TypeError("Results must be a list")

        for i, result in enumerate(results):
            if not isinstance(result, RawResult):
                raise TypeError(f"Result {i} must be RawResult instance, " f"got {type(result)}")

            if not result.text or not result.text.strip():
                raise ValueError(f"Result {i} text cannot be empty")

    def _filter_by_threshold(self, ranked_results: list[RankedResult]) -> list[RankedResult]:
        """Filter results by score threshold.

        Args:
            ranked_results: Results with relevance scores

        Returns:
            Filtered results above threshold
        """
        if self.config.score_threshold <= 0.0:
            return ranked_results

        filtered = [r for r in ranked_results if r.relevance_score >= self.config.score_threshold]

        logger.debug(
            f"Filtered {len(ranked_results)} results to {len(filtered)} "
            f"above threshold {self.config.score_threshold}"
        )

        return filtered

    def _apply_top_k(
        self, ranked_results: list[RankedResult], top_k: int | None = None
    ) -> list[RankedResult]:
        """Apply top-k filtering to results.

        Args:
            ranked_results: Results to filter
            top_k: Number of top results (uses config default if None)

        Returns:
            Top-k results
        """
        k = top_k or self.config.top_k

        if k <= 0:
            raise ValueError(f"top_k must be positive, got {k}")

        limited = ranked_results[:k]

        if len(limited) < len(ranked_results):
            logger.debug(f"Limited results from {len(ranked_results)} to top {k}")

        return limited

    def _update_ranks(self, ranked_results: list[RankedResult]) -> list[RankedResult]:
        """Update rank fields to reflect new ordering.

        Args:
            ranked_results: Results to update

        Returns:
            Results with updated new_rank fields
        """
        for new_rank, result in enumerate(ranked_results, start=1):
            result.new_rank = new_rank

        return ranked_results
