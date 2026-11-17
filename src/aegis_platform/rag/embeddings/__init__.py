"""Embedding providers for text-to-vector conversion.

This package contains abstract protocols, configuration classes, and concrete
implementations for generating embeddings from text using various providers.

Modules:
    base: Abstract EmbeddingProvider protocol (deprecated - use __init__ below)
    openai: OpenAI embedding provider implementation

Classes exported:
    EmbeddingConfig: Configuration for embedding provider instantiation
    EmbeddingProvider: Abstract protocol for embedding providers
    OpenAIEmbedding: OpenAI implementation
    OpenAIEmbeddingConfig: OpenAI-specific configuration

Usage:
    from aegis_platform.rag.embeddings import EmbeddingConfig, EmbeddingProvider
    from aegis_platform.rag.embeddings.openai import OpenAIEmbedding
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from aegis_platform.rag.embeddings.openai import (
    OpenAIEmbedding,
    OpenAIEmbeddingConfig,
)
from aegis_platform.rag.models import EmbeddingRequest, EmbeddingResponse


@dataclass
class EmbeddingConfig:
    """Configuration for embedding provider instantiation.

    Attributes:
        provider: Provider class name (e.g., 'openai', 'local', 'mock')
        model: Model name or identifier
        api_key: API key if required (from environment)
        base_url: Optional base URL for API endpoints
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        batch_size: Maximum batch size for batch operations
        cache_enabled: Whether to cache embeddings
    """

    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 100
    cache_enabled: bool = False


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Abstract protocol for embedding providers.

    Implementations must provide methods to:
    - Embed individual texts
    - Embed batches of texts efficiently
    - Report embedding dimensions
    - Handle errors gracefully

    This is intentionally abstract to allow multiple implementations:
    - OpenAI API (text-embedding-3-small, text-embedding-3-large)
    - Local models (sentence-transformers, FastEmbed)
    - Mock providers for testing
    - Custom enterprise providers

    Providers are instantiated via factory function with EmbeddingConfig.
    """

    async def embed_text(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embed a single text string.

        Args:
            request: Embedding request with text and optional model override

        Returns:
            EmbeddingResponse with embedding vector and metadata

        Raises:
            ValueError: If text is empty or malformed
            RuntimeError: If embedding fails (connection, quota, etc.)
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResponse]:
        """Embed multiple texts efficiently.

        Args:
            texts: List of text strings to embed

        Returns:
            List of EmbeddingResponse objects in same order as input

        Raises:
            ValueError: If texts list is empty or exceeds batch limits
            RuntimeError: If embedding fails
        """
        ...

    async def get_dimension(self) -> int:
        """Get embedding vector dimension for this provider.

        Returns:
            Number of dimensions in embedding vectors

        Note:
            Dimension is model-specific and fixed for a provider.
            Must be known before inserting vectors into storage.
        """
        ...

    async def health_check(self) -> bool:
        """Check if provider is available and responding.

        Returns:
            True if provider is healthy, False otherwise

        Note:
            Used for startup validation and monitoring.
        """
        ...


__all__ = [
    "EmbeddingConfig",
    "EmbeddingProvider",
    "OpenAIEmbedding",
    "OpenAIEmbeddingConfig",
]
