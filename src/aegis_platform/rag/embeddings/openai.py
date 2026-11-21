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
from typing import Any

from openai import APIConnectionError, AsyncOpenAI, RateLimitError

from aegis_platform.rag.models import EmbeddingRequest, EmbeddingResponse


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

    Wraps OpenAI's embedding API with:
    - Async/await for non-blocking operations
    - Configurable batching respecting API limits
    - Automatic retries with exponential backoff
    - Rate limit error handling
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
        self.client = AsyncOpenAI(api_key=config.api_key)

    async def embed_text(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embed a single text string.

        Args:
            request: Embedding request with text and optional model override

        Returns:
            EmbeddingResponse with embedding vector and metadata

        Raises:
            ValueError: If text is empty or malformed
            RuntimeError: If embedding fails after retries
        """
        if not request.text or not request.text.strip():
            raise ValueError("Cannot embed empty text")

        text = request.text.strip()
        model = request.model or self.config.model

        response = await self._embed_with_retry([text], model)

        if not response:
            raise RuntimeError("Failed to generate embedding")

        embedding_data = response[0]

        return EmbeddingResponse(
            text=text,
            embedding=embedding_data["embedding"],
            model=model,
            dimension=len(embedding_data["embedding"]),
            tokens_used=embedding_data.get("tokens_used", 0),
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResponse]:
        """Embed multiple texts efficiently.

        Args:
            texts: List of text strings to embed

        Returns:
            List of EmbeddingResponse objects in same order as input

        Raises:
            ValueError: If texts list is empty or exceeds batch limits
            RuntimeError: If embedding fails after retries
        """
        if not texts:
            raise ValueError("Cannot embed empty text list")

        if len(texts) > self.config.batch_size:
            raise ValueError(f"Batch size {len(texts)} exceeds limit {self.config.batch_size}")

        clean_texts = [text.strip() for text in texts if text and text.strip()]
        if not clean_texts:
            raise ValueError("All texts in batch are empty")

        embeddings_data = await self._embed_with_retry(clean_texts, self.config.model)

        return [
            EmbeddingResponse(
                text=clean_texts[i],
                embedding=data["embedding"],
                model=self.config.model,
                dimension=len(data["embedding"]),
                tokens_used=data.get("tokens_used", 0),
            )
            for i, data in enumerate(embeddings_data)
        ]

    async def get_dimension(self) -> int:
        """Get embedding vector dimension for this provider.

        Returns:
            Number of dimensions in embedding vectors (always 1536)

        Note:
            text-embedding-3-small model produces fixed 1536-dimensional vectors
        """
        return self.config.dimension

    async def health_check(self) -> bool:
        """Check if provider is available and responding.

        Returns:
            True if provider is healthy, False otherwise

        Note:
            Performs a simple embedding call to verify API connectivity
        """
        try:
            response = await self._embed_with_retry(["health"], self.config.model)
            return bool(response)
        except Exception:  # noqa: BLE001
            return False

    async def _embed_with_retry(self, texts: list[str], model: str) -> list[dict[str, Any]]:
        """Internal method to embed texts with retry logic.

        Args:
            texts: List of texts to embed
            model: Model to use for embedding

        Returns:
            List of embedding data dictionaries

        Raises:
            RuntimeError: If all retries exhausted
        """
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.embeddings.create(
                    input=texts,
                    model=model,
                )

                return [
                    {
                        "embedding": item.embedding,
                        "tokens_used": (
                            response.usage.prompt_tokens if response.usage is not None else 0
                        ),
                    }
                    for item in response.data
                ]

            except RateLimitError as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                continue

            except APIConnectionError as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                continue

            except Exception as e:
                raise RuntimeError(f"Embedding failed: {e!s}") from e

        raise RuntimeError(f"Failed to embed after {self.config.max_retries} retries: {last_error}")
