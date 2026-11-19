"""Cross-encoder based result reranking for RAG systems.

This module implements reranking using cross-encoder models from
sentence-transformers, which directly score query-document relevance
without requiring separate embeddings.

Features:
- Cross-encoder model support (Hugging Face)
- Batch inference for efficiency
- Configurable models and device
- Lazy model loading
- Singleton pattern for resource efficiency
- Async support

Example:
    >>> from aegis_platform.rag.cross_encoder_reranker import (
    ...     get_cross_encoder_reranker
    ... )
    >>>
    >>> reranker = get_cross_encoder_reranker()
    >>> reranked = await reranker.rerank(
    ...     query="what is RAG?",
    ...     results=search_results,
    ...     top_k=10
    ... )
"""

import logging
import math
from typing import Any

from pydantic import Field

from aegis_platform.rag.reranker_base import (
    RankedResult,
    RankerBase,
    RankerStrategy,
    RawResult,
    RerankerConfig,
)

logger = logging.getLogger(__name__)


class CrossEncoderConfig(RerankerConfig):
    """Configuration specific to cross-encoder reranking."""

    model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Hugging Face model name for cross-encoder",
    )
    device: str = Field(default="cpu", description="Device for inference (cpu, cuda, mps)")
    max_length: int = Field(default=512, description="Maximum sequence length for model")
    trust_remote_code: bool = Field(
        default=False, description="Trust remote code from Hugging Face"
    )
    cache_model: bool = Field(default=True, description="Cache model in memory after first load")

    def __init__(self, **data: Any) -> None:
        """Initialize cross-encoder config."""
        super().__init__(**data)
        self.strategy = RankerStrategy.DENSE


class CrossEncoderReranker(RankerBase):
    """Reranker using cross-encoder models for query-document scoring.

    Cross-encoders take query and document as input and directly score
    their relevance, unlike bi-encoders which embed separately.

    Attributes:
        config: Cross-encoder configuration
        model: Lazy-loaded cross-encoder model
    """

    def __init__(self, config: CrossEncoderConfig | None = None) -> None:
        """Initialize cross-encoder reranker.

        Args:
            config: Cross-encoder configuration
        """
        if config is None:
            config = CrossEncoderConfig()

        super().__init__(config)

        self.config: CrossEncoderConfig = config
        self.model: Any = None
        self._model_loaded = False
        self._apply_sigmoid = False

        logger.debug(
            f"Cross-encoder reranker initialized: "
            f"model={self.config.model_name}, device={self.config.device}"
        )

    def _ensure_model_loaded(self) -> None:
        """Load cross-encoder model if not already loaded.

        Uses lazy loading to defer model download until first use.

        Raises:
            ImportError: If sentence-transformers not installed
            RuntimeError: If model loading fails
        """
        if self._model_loaded and self.model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as e:
            msg = (
                "sentence-transformers required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            )
            logger.error(msg)
            raise ImportError(msg) from e

        try:
            self.model = CrossEncoder(
                self.config.model_name,
                device=self.config.device,
                max_length=self.config.max_length,
                trust_remote_code=self.config.trust_remote_code,
            )
            self._model_loaded = True
            # ms-marco cross-encoders emit raw logits (Identity activation); clamping those
            # to [0, 1] collapses every negative score to 0.0, so map them with a sigmoid.
            activation = getattr(self.model, "activation_fn", None)
            self._apply_sigmoid = type(activation).__name__ == "Identity"
            logger.debug(f"Cross-encoder model loaded: {self.config.model_name}")
        except Exception as e:
            msg = f"Failed to load cross-encoder model: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    def _normalize_score(self, raw: float) -> float:
        """Map a raw model score into [0, 1] when normalization is enabled."""
        if not self.config.normalize_scores:
            return raw
        if self._apply_sigmoid:
            return 1.0 / (1.0 + math.exp(-raw))
        return max(0.0, min(1.0, raw))

    async def score(self, query: str, document: str) -> float:
        """Score query-document pair relevance.

        Args:
            query: Query string
            document: Document string

        Returns:
            Relevance score (0-1 range)

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If scoring fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not document or not document.strip():
            raise ValueError("Document cannot be empty")

        self._ensure_model_loaded()

        try:
            score = self.model.predict([[query, document]], show_progress_bar=False)

            normalized_score = self._normalize_score(float(score[0]))

            logger.debug(f"Scored query-document pair: {normalized_score:.4f}")

            return normalized_score
        except Exception as e:
            msg = f"Failed to score query-document pair: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    async def batch_score(self, query: str, documents: list[str]) -> list[float]:
        """Score multiple documents for a single query (vectorized).

        Uses batch inference for better performance than sequential scoring.

        Args:
            query: Query string
            documents: List of document strings

        Returns:
            List of relevance scores (one per document)

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If scoring fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not documents:
            return []

        if not all(isinstance(doc, str) and doc.strip() for doc in documents):
            raise ValueError("All documents must be non-empty strings")

        self._ensure_model_loaded()

        try:
            pairs = [[query, doc] for doc in documents]

            scores = self.model.predict(
                pairs, batch_size=self.config.batch_size, show_progress_bar=False
            )

            normalized_scores = [self._normalize_score(float(score)) for score in scores]

            logger.debug(
                f"Batch scored {len(documents)} documents: "
                f"mean={sum(normalized_scores) / len(normalized_scores):.4f}"
            )

            return normalized_scores
        except Exception as e:
            msg = f"Failed to batch score documents: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    async def rerank(
        self, query: str, results: list[RawResult], top_k: int | None = None
    ) -> list[RankedResult]:
        """Rerank results using cross-encoder model.

        Args:
            query: Search query
            results: Raw search results
            top_k: Number of top results to return

        Returns:
            Reranked results sorted by relevance (descending)

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If reranking fails
        """
        self._validate_inputs(query, results)

        try:
            documents = [r.text for r in results]

            scores = await self.batch_score(query, documents)

            ranked_results = []
            for i, (original_result, score) in enumerate(zip(results, scores, strict=True)):
                ranked = RankedResult(
                    result_id=original_result.result_id,
                    text=original_result.text,
                    relevance_score=score,
                    source_id=original_result.source_id,
                    metadata=original_result.metadata.copy() if self.config.return_metadata else {},
                    original_rank=i + 1,
                    new_rank=0,
                )
                ranked_results.append(ranked)

            ranked_results.sort(key=lambda x: x.relevance_score, reverse=True)

            filtered = self._filter_by_threshold(ranked_results)

            limited = self._apply_top_k(filtered, top_k)

            final = self._update_ranks(limited)

            logger.info(
                f"Reranked {len(results)} results to {len(final)} "
                f"(top-k={top_k or self.config.top_k})"
            )

            return final
        except ValueError as e:
            logger.error(f"Reranking validation error: {e}")
            raise
        except Exception as e:
            msg = f"Reranking failed: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    def reset_model(self) -> None:
        """Reset cached model to free memory.

        Useful for cleanup or model reload.
        """
        if self.model is not None:
            del self.model
            self.model = None
            self._model_loaded = False
            logger.debug("Cross-encoder model reset")


_cross_encoder_instance: CrossEncoderReranker | None = None


def get_cross_encoder_reranker(config: CrossEncoderConfig | None = None) -> CrossEncoderReranker:
    """Get or create singleton cross-encoder reranker instance.

    Uses lazy initialization pattern for resource efficiency.

    Args:
        config: Configuration (only used on first call)

    Returns:
        Singleton CrossEncoderReranker instance

    Example:
        >>> reranker = get_cross_encoder_reranker()
        >>> reranked = await reranker.rerank(query, results)
    """
    global _cross_encoder_instance

    if _cross_encoder_instance is None:
        _cross_encoder_instance = CrossEncoderReranker(config)
        logger.debug("Created cross-encoder reranker singleton")

    return _cross_encoder_instance


def reset_cross_encoder_instance() -> None:
    """Reset singleton instance for testing.

    Clears the cached model and instance.
    """
    global _cross_encoder_instance

    if _cross_encoder_instance is not None:
        _cross_encoder_instance.reset_model()
        _cross_encoder_instance = None
        logger.debug("Reset cross-encoder reranker singleton")
