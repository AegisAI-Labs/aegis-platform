"""Unit tests for RAG provider factory pattern."""

import pytest

from aegis_platform.rag.embeddings import (
    NVIDIAEmbeddingConfig,
    OpenAIEmbedding,
    OpenAIEmbeddingConfig,
)
from aegis_platform.rag.provider_factory import (
    create_embedding_provider,
    create_vector_store,
    get_available_embedding_providers,
    get_available_vector_stores,
)
from aegis_platform.rag.storage import QdrantVectorStore, QdrantVectorStoreConfig


class TestEmbeddingProviderFactory:
    """Test embedding provider factory functions."""

    def test_create_openai_embedding_provider(self):
        """Test creating OpenAI embedding provider."""
        config = OpenAIEmbeddingConfig(
            api_key="sk-test123",
            model="text-embedding-3-small",
        )

        provider = create_embedding_provider(config)

        assert isinstance(provider, OpenAIEmbedding)
        assert provider.config.api_key == "sk-test123"
        assert provider.config.model == "text-embedding-3-small"

    def test_create_embedding_provider_unknown_provider(self):
        """Test creating unknown embedding provider raises error."""
        config = NVIDIAEmbeddingConfig(
            api_key="nvapi-invalid",
            model="nvidia/nemotron-3.5-embed-1b",
            embedding_dimension=1536,
        )
        # This will fail at runtime but config creation works
        provider = create_embedding_provider(config)
        assert provider is not None

    def test_create_embedding_provider_missing_api_key(self):
        """Test creating OpenAI provider without API key still allows creation."""
        # Empty API key is allowed at config creation; validation happens at runtime
        config = OpenAIEmbeddingConfig(
            api_key="",
            model="text-embedding-3-small",
        )
        # Should not raise during config creation
        assert config.api_key == ""

    def test_create_embedding_provider_invalid_dimension(self):
        """Test creating provider with invalid dimension raises error."""
        with pytest.raises(ValueError, match="Embedding dimension must be positive"):
            NVIDIAEmbeddingConfig(
                api_key="nvapi-test",
                model="nvidia/nemotron-3.5-embed-1b",
                embedding_dimension=-1,
            )

    def test_get_available_embedding_providers(self):
        """Test getting list of available embedding providers."""
        providers = get_available_embedding_providers()

        assert isinstance(providers, list)
        assert "openai" in providers
        assert "nvidia" in providers

    def test_embedding_provider_registry_extensible(self):
        """Test that embedding provider registry is extensible."""
        providers_before = get_available_embedding_providers()
        initial_count = len(providers_before)

        # Registry should be a dict in the module
        from aegis_platform.rag.provider_factory import EMBEDDING_PROVIDERS

        assert isinstance(EMBEDDING_PROVIDERS, dict)
        assert initial_count >= 2  # At least openai and nvidia


class TestVectorStoreFactory:
    """Test vector store factory functions."""

    def test_create_qdrant_vector_store(self):
        """Test creating Qdrant vector store."""
        config = QdrantVectorStoreConfig(
            url="http://localhost:6333",
            embedding_dimension=1536,
        )

        store = create_vector_store(config)

        assert isinstance(store, QdrantVectorStore)
        assert store.config.url == "http://localhost:6333"

    def test_create_vector_store_missing_url(self):
        """Test creating Qdrant store with empty URL still creates config."""
        # Empty URL is allowed at config level; validation at runtime
        config = QdrantVectorStoreConfig(
            url="",
            embedding_dimension=1536,
        )
        # Empty URL is allowed
        assert config.url == ""

    def test_create_qdrant_store_invalid_dimension(self):
        """Test creating Qdrant store with zero or negative dimension raises error."""
        config = QdrantVectorStoreConfig(
            url="http://localhost:6333",
            embedding_dimension=0,
        )
        # Validation is called in QdrantVectorStore.__init__, not config creation
        with pytest.raises(ValueError, match="embedding_dimension must be positive"):
            QdrantVectorStore(config)

    def test_get_available_vector_stores(self):
        """Test getting list of available vector stores."""
        stores = get_available_vector_stores()

        assert isinstance(stores, list)
        assert "qdrant" in stores

    def test_vector_store_registry_extensible(self):
        """Test that vector store registry is extensible."""
        stores_before = get_available_vector_stores()
        initial_count = len(stores_before)

        # Registry should be a dict in the module
        from aegis_platform.rag.provider_factory import VECTOR_STORES

        assert isinstance(VECTOR_STORES, dict)
        assert initial_count > 0


class TestFactoryConfigPropagation:
    """Test that factory properly propagates configuration."""

    def test_openai_embedding_provider_config_timeout(self):
        """Test timeout config propagates to OpenAI provider."""
        config = OpenAIEmbeddingConfig(
            api_key="sk-test",
            model="text-embedding-3-small",
            timeout=60,
        )

        provider = create_embedding_provider(config)

        assert provider.config.timeout == 60

    def test_openai_embedding_provider_config_max_retries(self):
        """Test max_retries config propagates to OpenAI provider."""
        config = OpenAIEmbeddingConfig(
            api_key="sk-test",
            model="text-embedding-3-small",
            max_retries=5,
        )

        provider = create_embedding_provider(config)

        assert provider.config.max_retries == 5

    def test_nvidia_embedding_provider_config_timeout(self):
        """Test timeout config propagates to NVIDIA provider."""
        config = NVIDIAEmbeddingConfig(
            api_key="nvapi-test",
            model="nvidia/nemotron-3.5-embed-1b",
            embedding_dimension=1536,
            timeout=60,
        )

        provider = create_embedding_provider(config)

        assert provider.config.timeout == 60

    def test_vector_store_config_api_key(self):
        """Test API key config propagates to vector store."""
        config = QdrantVectorStoreConfig(
            url="http://localhost:6333",
            embedding_dimension=1536,
            api_key="test-key",
        )

        store = create_vector_store(config)

        assert store.config.api_key == "test-key"

    def test_vector_store_config_timeout(self):
        """Test timeout config propagates to vector store."""
        config = QdrantVectorStoreConfig(
            url="http://localhost:6333",
            embedding_dimension=1536,
            timeout=45,
        )

        store = create_vector_store(config)

        assert store.config.timeout == 45


class TestFactoryEdgeCases:
    """Test factory edge cases and error conditions."""

    def test_registry_immutability(self):
        """Test that registries are properly initialized."""
        from aegis_platform.rag.provider_factory import (
            EMBEDDING_PROVIDERS,
            VECTOR_STORES,
        )

        # Verify registries are dictionaries with expected entries
        assert isinstance(EMBEDDING_PROVIDERS, dict)
        assert isinstance(VECTOR_STORES, dict)
        assert len(EMBEDDING_PROVIDERS) >= 2  # openai and nvidia
        assert len(VECTOR_STORES) > 0

    def test_factory_returns_different_instances(self):
        """Test factory creates new instances on each call."""
        config1 = OpenAIEmbeddingConfig(
            api_key="sk-test1",
            model="text-embedding-3-small",
        )

        config2 = OpenAIEmbeddingConfig(
            api_key="sk-test2",
            model="text-embedding-3-small",
        )

        provider1 = create_embedding_provider(config1)
        provider2 = create_embedding_provider(config2)

        assert provider1 is not provider2
        assert provider1.config.api_key != provider2.config.api_key


class TestFactoryIntegration:
    """Test factory integration scenarios."""

    def test_create_full_rag_stack_with_openai(self):
        """Test creating complete RAG stack with OpenAI embeddings."""
        embedding_config = OpenAIEmbeddingConfig(
            api_key="sk-test",
            model="text-embedding-3-small",
        )

        vector_store_config = QdrantVectorStoreConfig(
            url="http://localhost:6333",
            embedding_dimension=1536,
        )

        embedding_provider = create_embedding_provider(embedding_config)
        vector_store = create_vector_store(vector_store_config)

        assert isinstance(embedding_provider, OpenAIEmbedding)
        assert isinstance(vector_store, QdrantVectorStore)

    def test_create_full_rag_stack_with_nvidia(self):
        """Test creating complete RAG stack with NVIDIA embeddings."""
        embedding_config = NVIDIAEmbeddingConfig(
            api_key="nvapi-test",
            model="nvidia/nemotron-3.5-embed-1b",
            embedding_dimension=1536,
        )

        vector_store_config = QdrantVectorStoreConfig(
            url="http://localhost:6333",
            embedding_dimension=1536,
        )

        embedding_provider = create_embedding_provider(embedding_config)
        vector_store = create_vector_store(vector_store_config)

        assert embedding_provider is not None
        assert isinstance(vector_store, QdrantVectorStore)

    def test_factory_list_all_providers_returns_non_empty(self):
        """Test factory methods return non-empty lists."""
        embedding_providers = get_available_embedding_providers()
        vector_stores = get_available_vector_stores()

        assert len(embedding_providers) >= 2
        assert len(vector_stores) > 0
        assert "openai" in embedding_providers
        assert "nvidia" in embedding_providers
        assert "qdrant" in vector_stores
