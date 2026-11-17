"""Factory functions for creating embedding providers and vector stores.

Provides factory functions that instantiate embedding providers and vector stores
based on configuration, enabling dependency injection and runtime provider selection
via environment variables.
"""

from aegis_platform.rag.embeddings import (
    EmbeddingConfig,
    EmbeddingProvider,
    OpenAIEmbedding,
    OpenAIEmbeddingConfig,
)
from aegis_platform.rag.storage import (
    QdrantVectorStore,
    QdrantVectorStoreConfig,
    VectorStore,
    VectorStoreConfig,
)

# Registry of available embedding providers
# Extensible for adding new providers (Claude, Anthropic, local models, etc.)
EMBEDDING_PROVIDERS = {
    "openai": OpenAIEmbedding,
    # Future: "claude": ClaudeEmbedding,
    # Future: "local": LocalEmbedding,
}

# Registry of available vector stores
# Extensible for adding new stores (Weaviate, Pinecone, Chroma, etc.)
VECTOR_STORES = {
    "qdrant": QdrantVectorStore,
    # Future: "weaviate": WeaviateStore,
    # Future: "pinecone": PineconeStore,
}


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Factory function to create embedding provider based on configuration.

    Creates an instance of an embedding provider from the registry.
    All providers implement the EmbeddingProvider protocol.

    Args:
        config: EmbeddingConfig with provider name and parameters

    Returns:
        Instantiated EmbeddingProvider implementation

    Raises:
        ValueError: If provider is not found or config is invalid

    Example:
        >>> from aegis_platform.rag import EmbeddingConfig, create_embedding_provider
        >>> config = EmbeddingConfig(
        ...     provider="openai",
        ...     api_key="sk-...",
        ...     model="text-embedding-3-small"
        ... )
        >>> embedder = create_embedding_provider(config)
        >>> response = await embedder.embed_text(EmbeddingRequest("text"))

    Available Providers:
        - openai: OpenAI text-embedding-3-small (1536 dimensions)
    """
    provider_class = EMBEDDING_PROVIDERS.get(config.provider)

    if not provider_class:
        available = ", ".join(EMBEDDING_PROVIDERS.keys())
        raise ValueError(
            f"Unknown embedding provider: {config.provider}. " f"Available: {available}"
        )

    # Provider-specific configuration handling
    if config.provider == "openai":
        dimension = 1536  # OpenAI text-embedding-3-small default
        if isinstance(config.base_url, int):
            dimension = config.base_url
        provider_config = OpenAIEmbeddingConfig(
            api_key=config.api_key or "",
            model=config.model or "text-embedding-3-small",
            dimension=dimension,
            timeout=config.timeout,
            max_retries=config.max_retries,
            batch_size=config.batch_size,
        )
        return provider_class(provider_config)

    raise ValueError(f"Unknown provider implementation: {config.provider}")


def create_vector_store(config: VectorStoreConfig) -> VectorStore:
    """Factory function to create vector store based on configuration.

    Creates an instance of a vector store from the registry.
    All stores implement the VectorStore protocol.

    Args:
        config: VectorStoreConfig with store type and parameters

    Returns:
        Instantiated VectorStore implementation

    Raises:
        ValueError: If store is not found or config is invalid

    Example:
        >>> from aegis_platform.rag import VectorStoreConfig, create_vector_store
        >>> config = VectorStoreConfig(
        ...     store_type="qdrant",
        ...     url="http://localhost:6333",
        ...     embedding_dimension=1536
        ... )
        >>> store = create_vector_store(config)
        >>> ids = await store.insert("docs", vectors, metadata)

    Available Stores:
        - qdrant: Qdrant (open-source, self-hosted)
    """
    store_class = VECTOR_STORES.get(config.store_type)

    if not store_class:
        available = ", ".join(VECTOR_STORES.keys())
        raise ValueError(f"Unknown vector store: {config.store_type}. " f"Available: {available}")

    # Store-specific configuration handling
    if config.store_type == "qdrant":
        qdrant_config = QdrantVectorStoreConfig(
            url=config.url or "http://localhost:6333",
            api_key=config.api_key,
            timeout=config.timeout,
            embedding_dimension=config.dimension,
            distance_metric=config.distance_metric,
        )
        return store_class(qdrant_config)

    raise ValueError(f"Unknown store implementation: {config.store_type}")


def get_available_embedding_providers() -> list[str]:
    """Get list of available embedding provider names.

    Returns:
        List of provider names that can be used with create_embedding_provider()
    """
    return list(EMBEDDING_PROVIDERS.keys())


def get_available_vector_stores() -> list[str]:
    """Get list of available vector store names.

    Returns:
        List of store names that can be used with create_vector_store()
    """
    return list(VECTOR_STORES.keys())
