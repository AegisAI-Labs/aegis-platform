"""OpenAI EmbeddingProvider implementation using text-embedding-3-small model.

This module provides a concrete implementation of the EmbeddingProvider protocol
using OpenAI's embedding API. It supports:
- Single and batch embedding with configurable batch size
- Automatic retries with exponential backoff for transient errors
- Rate limit handling and timeout configuration
- Health checks for connection validation

The embedding dimension is fixed at 1536 for text-embedding-3-small model.
"""

import asyncio
from dataclasses import dataclass

import numpy as np
from openai import APIConnectionError, AsyncOpenAI, RateLimitError

from aegis_platform.rag.embeddings.usage import sum_usage
from aegis_platform.rag.models import (
    PASSAGE,
    QUERY,
    EmbeddingRequest,
    EmbeddingResponse,
)

PROVIDER_NAME = "openai"


@dataclass(frozen=True)
class _Batch:
    """Result of one embeddings API call."""

    vectors: list[np.ndarray]
    input_tokens: int | None
    model: str | None


@dataclass
class OpenAIEmbeddingConfig:
    """Configuration for OpenAI EmbeddingProvider.

    Attributes:
        api_key: OpenAI API key (required)
        model: Model name (default: text-embedding-3-small)
        dimension: Embedding dimension (fixed at 1536 for this model)
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts for transient errors
        batch_size: Maximum batch size for batch operations (OpenAI limit: 2048)
    """

    api_key: str
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 100

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If parameters are invalid
        """
        if not self.api_key or not self.api_key.strip():
            raise ValueError("api_key cannot be empty")
        if self.batch_size <= 0 or self.batch_size > 2048:
            raise ValueError("batch_size must be between 1 and 2048")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.dimension != 1536:
            raise ValueError(
                f"text-embedding-3-small always returns 1536 dimensions, got {self.dimension}"
            )


class OpenAIEmbedding:
    """OpenAI EmbeddingProvider implementation.

    Conforms to ``EmbeddingProvider`` (``embed_text(str) -> np.ndarray``,
    ``embed_batch(list[str]) -> list[np.ndarray]``), which is what ``RAGService`` calls.

    - Async/await for non-blocking operations
    - Inputs larger than ``batch_size`` are split into several API calls
    - Output order and length always match the input (empty texts are rejected, never dropped)
    - Bounded retries for rate-limit and connection errors (the SDK's own retries are disabled)
    """

    def __init__(self, config: OpenAIEmbeddingConfig) -> None:
        """Initialize OpenAI embedding provider.

        Args:
            config: Configuration with API key and parameters

        Raises:
            ValueError: If configuration is invalid
        """
        config.validate()
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            timeout=config.timeout,
            # Retries are handled once, in _embed_with_retry.
            max_retries=0,
        )

    @property
    def model(self) -> str:
        return self.config.model

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embed ``request.texts`` (split into API-sized batches), with summed token usage.

        ``input_type`` is ignored: OpenAI embedding models are symmetric.

        Raises:
            ValueError: If the request is empty or any text is empty
            RuntimeError: If embedding fails after retries
        """
        texts = request.texts
        if not texts:
            raise ValueError("Cannot embed empty text list")

        empty = [index for index, text in enumerate(texts) if not text or not text.strip()]
        if empty:
            raise ValueError(f"Cannot embed empty texts at positions {empty[:10]}")

        clean_texts = [text.strip() for text in texts]
        embeddings: list[np.ndarray] = []
        token_counts: list[int | None] = []
        served_model = self.config.model

        for start in range(0, len(clean_texts), self.config.batch_size):
            batch = await self._embed_with_retry(
                clean_texts[start : start + self.config.batch_size], self.config.model
            )
            embeddings.extend(batch.vectors)
            token_counts.append(batch.input_tokens)
            served_model = batch.model or served_model

        return EmbeddingResponse(
            embeddings=embeddings,
            model=served_model,
            provider=PROVIDER_NAME,
            dimension=len(embeddings[0]),
            usage=sum_usage(token_counts),
        )

    async def embed_text(self, text: str) -> np.ndarray:
        """Embed a single query text.

        Raises:
            ValueError: If text is empty or whitespace
            RuntimeError: If embedding fails after retries
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        response = await self.embed(EmbeddingRequest(texts=[text], input_type=QUERY))
        return response.embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed many passages; result ``i`` is the embedding of ``texts[i]``.

        Raises:
            ValueError: If the list is empty or any text is empty
            RuntimeError: If embedding fails after retries
        """
        response = await self.embed(EmbeddingRequest(texts=texts, input_type=PASSAGE))
        return response.embeddings

    async def get_dimension(self) -> int:
        """Embedding dimension (1536 for text-embedding-3-small)."""
        return self.config.dimension

    async def health_check(self) -> bool:
        """Check if provider is available and responding."""
        try:
            batch = await self._embed_with_retry(["health"], self.config.model)
            return bool(batch.vectors)
        except Exception:  # noqa: BLE001
            return False

    async def _embed_with_retry(self, texts: list[str], model: str) -> _Batch:
        """Embed one API batch with bounded retries.

        Raises:
            RuntimeError: If the request fails or all retries are exhausted. The message
                contains the error type only, never provider response text.
        """
        attempts = max(self.config.max_retries, 1)
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                response = await self.client.embeddings.create(
                    input=texts,
                    model=model,
                )
            except RateLimitError as e:
                if _is_quota_exhausted(e):
                    # Billing/quota errors never succeed on retry.
                    raise RuntimeError("Embedding quota exhausted (insufficient_quota)") from e
                last_error = e
                if attempt < attempts - 1:
                    await asyncio.sleep(2**attempt)
                continue
            except APIConnectionError as e:
                last_error = e
                if attempt < attempts - 1:
                    await asyncio.sleep(2**attempt)
                continue
            except Exception as e:
                raise RuntimeError(f"Embedding failed ({type(e).__name__})") from e

            # The API may return items out of order; sort by index to keep alignment.
            items = sorted(response.data, key=lambda item: item.index)
            if len(items) != len(texts):
                raise RuntimeError(
                    f"Embedding API returned {len(items)} vectors for {len(texts)} inputs"
                )
            usage = getattr(response, "usage", None)
            tokens = getattr(usage, "prompt_tokens", None)
            served_model = getattr(response, "model", None)
            return _Batch(
                vectors=[np.asarray(item.embedding, dtype=np.float32) for item in items],
                input_tokens=tokens if isinstance(tokens, int) else None,
                model=served_model if isinstance(served_model, str) and served_model else None,
            )

        raise RuntimeError(
            f"Failed to embed after {attempts} attempts ({type(last_error).__name__})"
        ) from last_error


def _is_quota_exhausted(error: RateLimitError) -> bool:
    body = error.body if isinstance(error.body, dict) else {}
    raw_nested = body.get("error")
    nested = raw_nested if isinstance(raw_nested, dict) else {}
    values = {body.get("type"), body.get("code"), nested.get("type"), nested.get("code")}
    return bool(values & {"insufficient_quota", "credit_balance_exhausted"})
