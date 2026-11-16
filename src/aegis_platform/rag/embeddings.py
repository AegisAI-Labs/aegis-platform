"""Abstract embedding provider protocol with configuration.

The EmbeddingProvider is designed as an abstract protocol to enable
runtime provider switching via dependency injection. Concrete implementations
(e.g., OpenAI, local models, mock providers) are instantiated based on
configuration from environment variables.

This follows the provider pattern used throughout AEGIS Platform for
LLM, reliability, and observability providers.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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
