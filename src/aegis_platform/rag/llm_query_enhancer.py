"""LLM-based query enhancement implementation for RAG systems.

This module provides an LLM-powered query enhancer that uses language models
to improve query quality through expansion and rewriting.

Features:
- Query expansion using LLM
- Query rewriting for clarity
- Hybrid strategy combining both
- Configurable LLM provider
- Fallback handling
- Result caching
- Singleton pattern for efficiency

Example:
    >>> from aegis_platform.rag.llm_query_enhancer import (
    ...     get_llm_query_enhancer, LLMQueryEnhancerConfig
    ... )
    >>>
    >>> # Get singleton enhancer
    >>> enhancer = get_llm_query_enhancer()
    >>>
    >>> # Enhance query
    >>> result = await enhancer.enhance(
    ...     query="What is RAG?",
    ...     strategy=EnhancementStrategy.EXPANSION
    ... )
    >>> print(result.expanded_queries)
    ['retrieval augmented generation', 'rag architecture', ...]
"""

import asyncio
import logging
from typing import Any

from pydantic import ConfigDict, Field

from aegis_platform.rag.query_enhancer_base import (
    EnhancedQuery,
    EnhancementStrategy,
    QueryEnhancerBase,
    QueryEnhancerConfig,
)

logger = logging.getLogger(__name__)


class LLMQueryEnhancerConfig(QueryEnhancerConfig):
    """Configuration for LLM-based query enhancement."""

    model_config = ConfigDict(validate_assignment=True)

    llm_provider: str = Field(default="openai", description="LLM provider (openai, anthropic, etc)")
    model_name: str = Field(default="gpt-3.5-turbo", description="LLM model to use")
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for generation (0=deterministic, 2=creative)",
    )
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    max_tokens: int = Field(default=256, ge=10, description="Max tokens in response")
    enable_caching: bool = Field(default=True, description="Enable LLM response caching")
    cache_ttl_seconds: int = Field(
        default=86400, ge=60, description="Cache time-to-live in seconds"
    )
    batch_size: int = Field(default=5, ge=1, description="Batch size for batch operations")


class LLMQueryEnhancer(QueryEnhancerBase):
    """LLM-based implementation of query enhancement.

    Uses a language model to generate query expansions and rewrites
    for improved search quality.
    """

    def __init__(self, config: LLMQueryEnhancerConfig | None = None) -> None:
        """Initialize LLM query enhancer.

        Args:
            config: LLM query enhancer configuration
        """
        self.config: LLMQueryEnhancerConfig = config or LLMQueryEnhancerConfig()
        super().__init__(self.config)

        self._cache: dict[str, Any] = {}
        self._llm_client: Any = None

        logger.debug(
            f"LLM query enhancer initialized with model={self.config.model_name}, "
            f"provider={self.config.llm_provider}, "
            f"temperature={self.config.temperature}"
        )

    async def _ensure_llm_client(self) -> None:
        """Initialize LLM client on first use (lazy loading).

        Raises:
            RuntimeError: If LLM client initialization fails
        """
        if self._llm_client is not None:
            return

        try:
            if self.config.llm_provider == "openai":
                from openai import AsyncOpenAI

                self._llm_client = AsyncOpenAI()
            elif self.config.llm_provider == "anthropic":
                from anthropic import AsyncAnthropic  # type: ignore

                self._llm_client = AsyncAnthropic()
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

            logger.debug(f"LLM client initialized for {self.config.llm_provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise RuntimeError(f"Failed to initialize LLM client: {e}") from e

    def _get_cache_key(self, query: str, operation: str) -> str:
        """Generate cache key for query operation.

        Args:
            query: Query string
            operation: Operation type (expand, rewrite, etc)

        Returns:
            Cache key
        """
        return f"{operation}:{query.lower()}"

    def _get_cached_result(self, cache_key: str) -> Any:
        """Get result from cache if available.

        Args:
            cache_key: Cache key

        Returns:
            Cached result or None if not found/expired
        """
        if not self.config.cache_enabled:
            return None

        result = self._cache.get(cache_key)
        if result is not None:
            logger.debug(f"Cache hit for {cache_key}")
            return result

        return None

    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Store result in cache.

        Args:
            cache_key: Cache key
            result: Result to cache
        """
        if self.config.cache_enabled:
            self._cache[cache_key] = result
            logger.debug(f"Cached result for {cache_key}")

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with given prompt.

        Args:
            prompt: Prompt to send to LLM

        Returns:
            LLM response text

        Raises:
            RuntimeError: If LLM call fails
        """
        await self._ensure_llm_client()

        try:
            if self.config.llm_provider == "openai":
                response = await self._llm_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout_seconds,
                )
                return response.choices[0].message.content

            elif self.config.llm_provider == "anthropic":
                response = await self._llm_client.messages.create(
                    model=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    timeout=self.config.timeout_seconds,
                )
                return response.content[0].text

            else:
                raise ValueError(f"Unsupported provider: {self.config.llm_provider}")

        except TimeoutError as e:
            logger.error(f"LLM call timeout: {e}")
            raise TimeoutError("LLM call timed out") from e
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"LLM call failed: {e}") from e

    def _parse_expansion_response(self, response: str) -> list[str]:
        """Parse expansion terms from LLM response.

        Args:
            response: LLM response text

        Returns:
            List of expansion terms
        """
        terms = []

        for line in response.strip().split("\n"):
            line = line.strip()

            if not line:
                continue

            # Remove common prefixes (1., -, *, etc)
            if line and line[0] in "0123456789.-*•":
                line = line.lstrip("0123456789.-*•").strip()

            if line:
                terms.append(line)

        return terms

    def _parse_rewrite_response(self, response: str) -> str:
        """Parse rewritten query from LLM response.

        Args:
            response: LLM response text

        Returns:
            Rewritten query
        """
        return response.strip()

    async def expand_query(self, query: str) -> list[str]:
        """Generate query expansions using LLM.

        Args:
            query: Query to expand

        Returns:
            List of expansion terms

        Raises:
            ValueError: If input is invalid
            RuntimeError: If expansion fails
        """
        self._validate_query(query)
        normalized = self._normalize_query(query)

        # Check cache
        cache_key = self._get_cache_key(normalized, "expand")
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            return cached

        try:
            prompt = (
                f"Generate 5-7 semantically related terms, synonyms, and "
                f"alternative phrasings for this query. Return one term per line.\n\n"
                f"Query: {query}\n\n"
                f"Related terms:"
            )

            response = await self._call_llm(prompt)
            terms = self._parse_expansion_response(response)

            # Filter and limit
            filtered = self._filter_expansions(terms)
            limited = self._limit_expansions(filtered)

            # Add original if configured
            if self.config.include_original and normalized not in limited:
                limited = [normalized] + limited

            self._cache_result(cache_key, limited)
            logger.debug(f"Expanded query '{query}' to {len(limited)} terms")

            return limited

        except (RuntimeError, ValueError) as e:
            logger.error(f"Query expansion failed: {e}")
            # Fallback: return original query
            fallback = [normalized] if self.config.include_original else []
            return fallback

    async def rewrite_query(self, query: str) -> str:
        """Rewrite query for improved clarity using LLM.

        Args:
            query: Query to rewrite

        Returns:
            Rewritten query

        Raises:
            ValueError: If input is invalid
            RuntimeError: If rewriting fails
        """
        self._validate_query(query)
        normalized = self._normalize_query(query)

        # Check cache
        cache_key = self._get_cache_key(normalized, "rewrite")
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            return cached

        try:
            prompt = (
                f"Rewrite this query to be more explicit, clear, and "
                f"optimized for semantic search. Maintain the original intent "
                f"but improve clarity and reduce ambiguity.\n\n"
                f"Original query: {query}\n\n"
                f"Rewritten query:"
            )

            response = await self._call_llm(prompt)
            rewritten = self._parse_rewrite_response(response)

            self._cache_result(cache_key, rewritten)
            logger.debug(f"Rewrote query '{query}' to '{rewritten}'")

            return rewritten

        except (RuntimeError, ValueError) as e:
            logger.error(f"Query rewriting failed: {e}")
            # Fallback: return normalized original
            return normalized

    async def enhance(
        self, query: str, strategy: EnhancementStrategy | None = None
    ) -> EnhancedQuery:
        """Enhance query using specified strategy.

        Args:
            query: Query to enhance
            strategy: Enhancement strategy (uses config default if None)

        Returns:
            EnhancedQuery with original + enhanced representations

        Raises:
            ValueError: If input is invalid
            RuntimeError: If enhancement fails
        """
        self._validate_query(query)

        strategy = strategy or self.config.strategy

        logger.debug(f"Enhancing query '{query}' with strategy={strategy}")

        try:
            if strategy == EnhancementStrategy.EXPANSION:
                expanded = await self.expand_query(query)
                return EnhancedQuery(
                    original_query=query,
                    expanded_queries=expanded,
                    expansion_terms=expanded,
                    rewritten_query=None,
                    strategy_used=strategy,
                    confidence_score=0.9,
                )

            elif strategy == EnhancementStrategy.REWRITE:
                rewritten = await self.rewrite_query(query)
                return EnhancedQuery(
                    original_query=query,
                    rewritten_query=rewritten,
                    strategy_used=strategy,
                    confidence_score=0.85,
                )

            elif strategy == EnhancementStrategy.HYBRID:
                # Both expansion and rewriting
                expanded_task = asyncio.create_task(self.expand_query(query))
                rewrite_task = asyncio.create_task(self.rewrite_query(query))

                expanded, rewritten = await asyncio.gather(expanded_task, rewrite_task)

                return EnhancedQuery(
                    original_query=query,
                    expanded_queries=expanded,
                    rewritten_query=rewritten,
                    expansion_terms=expanded,
                    strategy_used=strategy,
                    confidence_score=0.9,
                )

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        except Exception as e:
            logger.error(f"Query enhancement failed for '{query}': {e}")
            raise RuntimeError(f"Query enhancement failed: {e}") from e

    async def batch_enhance(
        self, queries: list[str], strategy: EnhancementStrategy | None = None
    ) -> list[EnhancedQuery]:
        """Enhance multiple queries with batching.

        Args:
            queries: List of queries to enhance
            strategy: Enhancement strategy (uses config default if None)

        Returns:
            List of EnhancedQuery objects

        Raises:
            ValueError: If inputs are invalid
        """
        if not queries:
            return []

        enhanced = []
        batch_size = self.config.batch_size

        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            tasks = [self.enhance(q, strategy) for q in batch]

            batch_results = await asyncio.gather(*tasks)
            enhanced.extend(batch_results)

        logger.debug(f"Batch enhanced {len(queries)} queries with strategy={strategy}")

        return enhanced


# Module-level singleton instance
_llm_query_enhancer_instance: LLMQueryEnhancer | None = None


def get_llm_query_enhancer(config: LLMQueryEnhancerConfig | None = None) -> LLMQueryEnhancer:
    """Get or create LLM query enhancer singleton.

    Args:
        config: Configuration (only used if creating new instance)

    Returns:
        LLM query enhancer instance
    """
    global _llm_query_enhancer_instance

    if _llm_query_enhancer_instance is None:
        _llm_query_enhancer_instance = LLMQueryEnhancer(config)
        logger.debug("Created new LLM query enhancer singleton")

    return _llm_query_enhancer_instance


def reset_llm_query_enhancer_instance() -> None:
    """Reset LLM query enhancer singleton (for testing).

    This clears the cached instance so a new one will be created on next call.
    """
    global _llm_query_enhancer_instance
    _llm_query_enhancer_instance = None
    logger.debug("Reset LLM query enhancer singleton")
