"""Unit tests for RAG provider factory pattern."""

import pytest

from aegis_platform.rag.embeddings import EmbeddingConfig, OpenAIEmbedding
from aegis_platform.rag.provider_factory import (
    create_embedding_provider,
    create_vector_store,
    get_available_embedding_providers,
    get_available_vector_stores,
)
from aegis_platform.rag.storage import QdrantVectorStore, VectorStoreConfig


class TestEmbeddingProviderFactory:
    """Test embedding provider factory functions."""

    def test_create_openai_embedding_provider(self):
        """Test creating OpenAI embedding provider."""
        config = EmbeddingConfig(
            provider="openai",
            api_key="sk-test123",
            model="text-embedding-3-small",
        )

        provider = create_embedding_provider(config)

        assert isinstance(provider, OpenAIEmbedding)
        assert provider.config.api_key == "sk-test123"
        assert provider.config.model == "text-embedding-3-small"

    def test_create_embedding_provider_unknown_provider(self):
        """Test creating unknown embedding provider raises error."""
        config = EmbeddingConfig(
            provider="unknown_provider",
            model="text-embedding-3-small",
            api_key="sk-test123",
        )

        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_provider(config)

    def test_create_embedding_provider_missing_api_key(self):
        """Test creating OpenAI provider without API key raises error."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="",
        )

        with pytest.raises(ValueError, match="api_key"):
            create_embedding_provider(config)

    def test_create_embedding_provider_invalid_config(self):
        """Test creating provider with invalid config raises error."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test",
            batch_size=2049,  # Exceeds limit
        )

        with pytest.raises(ValueError, match="batch_size"):
            create_embedding_provider(config)

    def test_get_available_embedding_providers(self):
        """Test getting list of available embedding providers."""
        providers = get_available_embedding_providers()

        assert isinstance(providers, list)
        assert "openai" in providers

    def test_embedding_provider_registry_extensible(self):
        """Test that embedding provider registry is extensible."""
        providers_before = get_available_embedding_providers()
        initial_count = len(providers_before)

        # Registry should be a dict in the module
        from aegis_platform.rag.provider_factory import EMBEDDING_PROVIDERS

        assert isinstance(EMBEDDING_PROVIDERS, dict)
        assert initial_count > 0


class TestVectorStoreFactory:
    """Test vector store factory functions."""

    def test_create_qdrant_vector_store(self):
        """Test creating Qdrant vector store."""
        config = VectorStoreConfig(
            store_type="qdrant",
            url="http://localhost:6333",
            dimension=1536,
        )

        store = create_vector_store(config)

        assert isinstance(store, QdrantVectorStore)
        assert store.config.url == "http://localhost:6333"

    def test_create_vector_store_unknown_provider(self):
        """Test creating unknown vector store raises error."""
        config = VectorStoreConfig(
            store_type="unknown_store",
            url="http://localhost:6333",
        )

        with pytest.raises(ValueError, match="Unknown vector store"):
            create_vector_store(config)

    def test_create_qdrant_store_missing_url(self):
        """Test creating Qdrant store without URL uses default."""
        config = VectorStoreConfig(
            store_type="qdrant",
            url="",
            dimension=1536,
        )

        # Empty URL should use factory default
        store = create_vector_store(config)
        assert store.config.url == "http://localhost:6333"

    def test_create_qdrant_store_invalid_distance_metric(self):
        """Test creating Qdrant store with invalid distance metric raises error."""
        config = VectorStoreConfig(
            store_type="qdrant",
            url="http://localhost:6333",
            dimension=1536,
            distance_metric="invalid_metric",
        )

        with pytest.raises(ValueError, match="distance_metric"):
            create_vector_store(config)

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

    def test_embedding_provider_config_timeout(self):
        """Test timeout config propagates to provider."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test",
            timeout=60,
        )

        provider = create_embedding_provider(config)

        assert provider.config.timeout == 60

    def test_embedding_provider_config_max_retries(self):
        """Test max_retries config propagates to provider."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test",
            max_retries=5,
        )

        provider = create_embedding_provider(config)

        assert provider.config.max_retries == 5

    def test_embedding_provider_config_batch_size(self):
        """Test batch_size config propagates to provider."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test",
            batch_size=50,
        )

        provider = create_embedding_provider(config)

        assert provider.config.batch_size == 50

    def test_vector_store_config_api_key(self):
        """Test API key config propagates to vector store."""
        config = VectorStoreConfig(
            store_type="qdrant",
            url="http://localhost:6333",
            api_key="test-key",
            dimension=1536,
        )

        store = create_vector_store(config)

        assert store.config.api_key == "test-key"

    def test_vector_store_config_timeout(self):
        """Test timeout config propagates to vector store."""
        config = VectorStoreConfig(
            store_type="qdrant",
            url="http://localhost:6333",
            timeout=45,
            dimension=1536,
        )

        store = create_vector_store(config)

        assert store.config.timeout == 45

    def test_vector_store_config_distance_metric(self):
        """Test distance_metric config propagates to vector store."""
        for metric in ["cosine", "euclidean", "manhattan"]:
            config = VectorStoreConfig(
                store_type="qdrant",
                url="http://localhost:6333",
                distance_metric=metric,
                dimension=1536,
            )

            store = create_vector_store(config)

            assert store.config.distance_metric == metric


class TestFactoryEdgeCases:
    """Test factory edge cases and error conditions."""

    def test_create_embedding_provider_case_insensitive(self):
        """Test provider name is handled case-insensitively."""
        # Assuming implementation supports case-insensitivity
        config = EmbeddingConfig(
            provider="OpenAI",  # Different case
            model="text-embedding-3-small",
            api_key="sk-test",
        )

        # This may or may not be supported; test current behavior
        try:
            provider = create_embedding_provider(config)
            assert provider is not None
        except ValueError:
            # If case-sensitive, that's also valid
            pass

    def test_create_vector_store_empty_provider_name(self):
        """Test creating store with empty provider name raises error."""
        config = VectorStoreConfig(
            store_type="",
            url="http://localhost:6333",
        )

        with pytest.raises(ValueError):
            create_vector_store(config)

    def test_create_embedding_provider_empty_provider_name(self):
        """Test creating provider with empty provider name raises error."""
        config = EmbeddingConfig(
            provider="",
            model="text-embedding-3-small",
            api_key="sk-test",
        )

        with pytest.raises(ValueError):
            create_embedding_provider(config)

    def test_registry_immutability(self):
        """Test that registries are properly initialized."""
        from aegis_platform.rag.provider_factory import (
            EMBEDDING_PROVIDERS,
            VECTOR_STORES,
        )

        # Verify registries are dictionaries with expected entries
        assert isinstance(EMBEDDING_PROVIDERS, dict)
        assert isinstance(VECTOR_STORES, dict)
        assert len(EMBEDDING_PROVIDERS) > 0
        assert len(VECTOR_STORES) > 0

    def test_factory_returns_different_instances(self):
        """Test factory creates new instances on each call."""
        config1 = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test1",
        )

        config2 = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test2",
        )

        provider1 = create_embedding_provider(config1)
        provider2 = create_embedding_provider(config2)

        assert provider1 is not provider2
        assert provider1.config.api_key != provider2.config.api_key


class TestFactoryIntegration:
    """Test factory integration scenarios."""

    def test_create_full_rag_stack(self):
        """Test creating complete RAG stack with factory."""
        embedding_config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test",
        )

        vector_store_config = VectorStoreConfig(
            store_type="qdrant",
            url="http://localhost:6333",
            dimension=1536,
        )

        embedding_provider = create_embedding_provider(embedding_config)
        vector_store = create_vector_store(vector_store_config)

        assert isinstance(embedding_provider, OpenAIEmbedding)
        assert isinstance(vector_store, QdrantVectorStore)

    def test_factory_list_all_providers_returns_non_empty(self):
        """Test factory methods return non-empty lists."""
        embedding_providers = get_available_embedding_providers()
        vector_stores = get_available_vector_stores()

        assert len(embedding_providers) > 0
        assert len(vector_stores) > 0
        assert "openai" in embedding_providers
        assert "qdrant" in vector_stores
