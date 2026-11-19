"""Base query enhancement interface for RAG systems.

This module provides abstract interfaces and common types for query
enhancement implementations that improve query quality before vector search.

Features:
- Abstract query enhancer base class
- Pluggable enhancement strategies (expansion, rewriting, hybrid)
- Configuration management
- Type-safe result models
- Async support

Example:
    >>> from aegis_platform.rag.query_enhancer_base import (
    ...     QueryEnhancerBase, EnhancedQuery, EnhancementStrategy
    ... )
    >>>
    >>> # Implement custom enhancer
    >>> class MyEnhancer(QueryEnhancerBase):
    ...     async def enhance(self, query: str, strategy=None):
    ...         # Custom enhancement logic
    ...         pass
    ...
    ...     async def expand_query(self, query: str) -> list[str]:
    ...         # Custom expansion logic
    ...         pass
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class EnhancementStrategy(str, Enum):
    """Available query enhancement strategies."""

    EXPANSION = "expansion"
    REWRITE = "rewrite"
    HYBRID = "hybrid"


class EnhancedQuery(BaseModel):
    """Enhanced query with multiple representations."""

    original_query: str = Field(..., description="Original user query")
    expanded_queries: list[str] = Field(
        default_factory=list, description="Query expansions (synonyms, related terms)"
    )
    rewritten_query: str | None = Field(None, description="Rewritten query for clarity")
    expansion_terms: list[str] = Field(
        default_factory=list, description="Extracted expansion terms"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Enhancement metadata (confidence, sources, etc)"
    )
    strategy_used: EnhancementStrategy = Field(..., description="Strategy used for enhancement")
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in enhancement quality"
    )


class QueryEnhancerConfig(BaseModel):
    """Configuration for query enhancement operations."""

    model_config = ConfigDict(validate_assignment=True)

    strategy: EnhancementStrategy = Field(
        default=EnhancementStrategy.EXPANSION, description="Default enhancement strategy to use"
    )
    max_expansions: int = Field(
        default=5, ge=1, description="Maximum number of expansions to generate"
    )
    min_expansion_length: int = Field(
        default=2, ge=1, description="Minimum length for expansion terms (chars)"
    )
    include_original: bool = Field(default=True, description="Include original query in expansions")
    normalize_query: bool = Field(default=True, description="Normalize query (lowercase, etc)")
    remove_stop_words: bool = Field(default=False, description="Remove stop words from query")
    timeout_seconds: float = Field(
        default=5.0, gt=0.0, description="Timeout for enhancement operation"
    )
    cache_enabled: bool = Field(default=True, description="Enable caching of enhancement results")


class QueryEnhancerBase(ABC):
    """Abstract base class for all query enhancement implementations.

    Defines the interface that all query enhancer implementations must follow.
    Query enhancers improve search quality by expanding or rewriting queries.
    """

    def __init__(self, config: QueryEnhancerConfig | None = None) -> None:
        """Initialize query enhancer with configuration.

        Args:
            config: Query enhancer configuration (uses defaults if None)
        """
        self.config = config or QueryEnhancerConfig()
        logger.debug(
            f"Query enhancer initialized with strategy={self.config.strategy}, "
            f"max_expansions={self.config.max_expansions}"
        )

    @abstractmethod
    async def enhance(
        self, query: str, strategy: EnhancementStrategy | None = None
    ) -> EnhancedQuery:
        """Enhance a query using specified strategy.

        Args:
            query: Query string to enhance
            strategy: Enhancement strategy (uses config default if None)

        Returns:
            EnhancedQuery with original + enhanced representations

        Raises:
            ValueError: If input is invalid
            RuntimeError: If enhancement fails
            TimeoutError: If operation exceeds timeout
        """

    @abstractmethod
    async def expand_query(self, query: str) -> list[str]:
        """Generate query expansions (synonyms, related terms).

        Args:
            query: Query string to expand

        Returns:
            List of expansion terms/queries

        Raises:
            ValueError: If input is invalid
            RuntimeError: If expansion fails
        """

    @abstractmethod
    async def rewrite_query(self, query: str) -> str:
        """Rewrite query for improved clarity and search quality.

        Args:
            query: Query string to rewrite

        Returns:
            Rewritten query

        Raises:
            ValueError: If input is invalid
            RuntimeError: If rewriting fails
        """

    async def batch_enhance(
        self, queries: list[str], strategy: EnhancementStrategy | None = None
    ) -> list[EnhancedQuery]:
        """Enhance multiple queries.

        Default implementation calls enhance() sequentially. Override for
        vectorized implementations.

        Args:
            queries: List of query strings
            strategy: Enhancement strategy (uses config default if None)

        Returns:
            List of EnhancedQuery objects

        Raises:
            ValueError: If inputs are invalid
        """
        if not queries:
            return []

        enhanced = []
        for query in queries:
            result = await self.enhance(query, strategy)
            enhanced.append(result)

        return enhanced

    def _validate_query(self, query: str) -> None:
        """Validate query input.

        Args:
            query: Query string to validate

        Raises:
            ValueError: If query is invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if len(query) > 10000:
            raise ValueError("Query exceeds maximum length (10000 chars)")

    def _normalize_query(self, query: str) -> str:
        """Normalize query string.

        Args:
            query: Query to normalize

        Returns:
            Normalized query
        """
        if self.config.normalize_query:
            query = query.lower().strip()

        return query

    def _deduplicate_terms(self, terms: list[str]) -> list[str]:
        """Remove duplicate terms preserving order.

        Args:
            terms: List of terms

        Returns:
            Deduplicated terms
        """
        seen = set()
        result = []

        for term in terms:
            if term not in seen:
                seen.add(term)
                result.append(term)

        return result

    def _filter_expansions(self, terms: list[str]) -> list[str]:
        """Filter expansion terms by length and other criteria.

        Args:
            terms: Expansion terms to filter

        Returns:
            Filtered terms
        """
        filtered = [t for t in terms if len(t) >= self.config.min_expansion_length]

        return self._deduplicate_terms(filtered)

    def _limit_expansions(self, terms: list[str]) -> list[str]:
        """Limit number of expansions.

        Args:
            terms: Expansion terms

        Returns:
            Limited to max_expansions count
        """
        return terms[: self.config.max_expansions]
