"""Factory functions for creating embedding providers and vector stores.

Provides factory functions that instantiate embedding providers and vector stores
based on configuration, enabling dependency injection and runtime provider selection
via environment variables.
"""

from aegis_platform.rag.embeddings import (
    EmbeddingProvider,
    NVIDIAEmbeddingConfig,
    NVIDIAEmbeddingProvider,
    OpenAIEmbedding,
    OpenAIEmbeddingConfig,
)
from aegis_platform.rag.storage import (
    QdrantVectorStore,
    QdrantVectorStoreConfig,
    VectorStore,
)

# Registry of available embedding providers
# Extensible for adding new providers (Claude, Cohere, local models, etc.)
EMBEDDING_PROVIDERS = {
    "openai": OpenAIEmbedding,
    "nvidia": NVIDIAEmbeddingProvider,
    # Future: "claude": ClaudeEmbeddingProvider,
    # Future: "cohere": CohereEmbeddingProvider,
    # Future: "local": LocalEmbeddingProvider,
}

# Registry of available vector stores
# Extensible for adding new stores (Weaviate, Pinecone, Chroma, etc.)
VECTOR_STORES = {
    "qdrant": QdrantVectorStore,
    # Future: "weaviate": WeaviateStore,
    # Future: "pinecone": PineconeStore,
}


def create_embedding_provider(
    config: OpenAIEmbeddingConfig | NVIDIAEmbeddingConfig,
) -> EmbeddingProvider:
    """Factory function to create embedding provider based on configuration.

    Creates an instance of an embedding provider from the registry.
    All providers implement the EmbeddingProvider protocol.

    Args:
        config: Configuration object (OpenAIEmbeddingConfig or NVIDIAEmbeddingConfig)

    Returns:
        Instantiated EmbeddingProvider implementation

    Raises:
        ValueError: If provider is not found or config is invalid
        TypeError: If config type doesn't match any provider

    Example:
        >>> from aegis_platform.rag.embeddings import OpenAIEmbeddingConfig
        >>> from aegis_platform.rag.provider_factory import create_embedding_provider
        >>> config = OpenAIEmbeddingConfig(api_key="sk-...")
        >>> embedder = create_embedding_provider(config)
        >>> embeddings = await embedder.embed_batch(["Hello", "World"])

    Available Providers:
        - openai: OpenAI text-embedding-3-small (1536 dimensions)
        - nvidia: NVIDIA Nemotron embeddings (1536 dimensions)
    """
    # Determine provider from config type
    if isinstance(config, OpenAIEmbeddingConfig):
        return OpenAIEmbedding(config)
    elif isinstance(config, NVIDIAEmbeddingConfig):
        return NVIDIAEmbeddingProvider(config)
    else:
        available = ", ".join(EMBEDDING_PROVIDERS.keys())
        raise TypeError(
            f"Unknown config type: {type(config).__name__}. Available providers: {available}"
        )


def create_vector_store(config: QdrantVectorStoreConfig) -> VectorStore:
    """Factory function to create vector store based on configuration.

    Creates an instance of a vector store from the registry.
    All stores implement the VectorStore protocol.

    Args:
        config: QdrantVectorStoreConfig with store parameters

    Returns:
        Instantiated VectorStore implementation

    Raises:
        ValueError: If store is not found or config is invalid
        TypeError: If config type doesn't match any store

    Example:
        >>> from aegis_platform.rag.storage import QdrantVectorStoreConfig
        >>> from aegis_platform.rag.provider_factory import create_vector_store
        >>> config = QdrantVectorStoreConfig(
        ...     url="https://...-qdrant.io",
        ...     api_key="...",
        ...     embedding_dimension=1536
        ... )
        >>> store = create_vector_store(config)
        >>> ids = await store.insert("docs", vectors, metadata)

    Available Stores:
        - qdrant: Qdrant (self-hosted, cloud, or in-memory)
    """
    if isinstance(config, QdrantVectorStoreConfig):
        return QdrantVectorStore(config)
    else:
        available = ", ".join(VECTOR_STORES.keys())
        raise TypeError(
            f"Unknown config type: {type(config).__name__}. Available stores: {available}"
        )


def get_available_embedding_providers() -> list[str]:
    """Get list of available embedding provider names.

    Returns:
        List of provider names that can be used with create_embedding_provider()

    Example:
        >>> providers = get_available_embedding_providers()
        >>> print(providers)  # ['openai', 'nvidia']
    """
    return list(EMBEDDING_PROVIDERS.keys())


def get_available_vector_stores() -> list[str]:
    """Get list of available vector store names.

    Returns:
        List of store names that can be used with create_vector_store()

    Example:
        >>> stores = get_available_vector_stores()
        >>> print(stores)  # ['qdrant']
    """
    return list(VECTOR_STORES.keys())
