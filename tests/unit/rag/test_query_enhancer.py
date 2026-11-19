"""Unit tests for query enhancement module.

Tests cover:
- Enhancement strategy enum
- EnhancedQuery model
- QueryEnhancerConfig configuration
- LLMQueryEnhancerConfig configuration
- LLM query enhancer functionality
- Query expansion
- Query rewriting
- Hybrid enhancement
- Edge cases (empty queries, long queries, special chars)
- Singleton pattern
- Caching behavior
"""

from unittest.mock import AsyncMock, patch

import pytest

from aegis_platform.rag.llm_query_enhancer import (
    LLMQueryEnhancer,
    LLMQueryEnhancerConfig,
    get_llm_query_enhancer,
    reset_llm_query_enhancer_instance,
)
from aegis_platform.rag.query_enhancer_base import (
    EnhancedQuery,
    EnhancementStrategy,
    QueryEnhancerConfig,
)


class TestEnhancementStrategy:
    """Test enhancement strategy enum."""

    def test_strategy_values(self) -> None:
        """Test enum has expected values."""
        assert EnhancementStrategy.EXPANSION.value == "expansion"
        assert EnhancementStrategy.REWRITE.value == "rewrite"
        assert EnhancementStrategy.HYBRID.value == "hybrid"

    def test_strategy_from_string(self) -> None:
        """Test creating strategy from string."""
        assert EnhancementStrategy("expansion") == EnhancementStrategy.EXPANSION
        assert EnhancementStrategy("rewrite") == EnhancementStrategy.REWRITE
        assert EnhancementStrategy("hybrid") == EnhancementStrategy.HYBRID


class TestEnhancedQuery:
    """Test EnhancedQuery model."""

    def test_create_basic_enhanced_query(self) -> None:
        """Test creating EnhancedQuery with minimal fields."""
        query = EnhancedQuery(
            original_query="What is RAG?", strategy_used=EnhancementStrategy.EXPANSION
        )

        assert query.original_query == "What is RAG?"
        assert query.strategy_used == EnhancementStrategy.EXPANSION
        assert query.expanded_queries == []
        assert query.rewritten_query is None
        assert query.confidence_score == 1.0

    def test_create_enhanced_query_with_expansions(self) -> None:
        """Test creating EnhancedQuery with expansions."""
        query = EnhancedQuery(
            original_query="What is RAG?",
            expanded_queries=["retrieval augmented generation", "rag architecture"],
            expansion_terms=["retrieval augmented generation", "rag architecture"],
            strategy_used=EnhancementStrategy.EXPANSION,
            confidence_score=0.95,
        )

        assert query.original_query == "What is RAG?"
        assert len(query.expanded_queries) == 2
        assert query.confidence_score == 0.95

    def test_create_enhanced_query_with_rewrite(self) -> None:
        """Test creating EnhancedQuery with rewritten query."""
        query = EnhancedQuery(
            original_query="ml models?",
            rewritten_query="What are machine learning models?",
            strategy_used=EnhancementStrategy.REWRITE,
            confidence_score=0.88,
        )

        assert query.original_query == "ml models?"
        assert query.rewritten_query == "What are machine learning models?"
        assert query.strategy_used == EnhancementStrategy.REWRITE

    def test_enhanced_query_confidence_validation(self) -> None:
        """Test confidence score validation."""
        with pytest.raises(ValueError):
            EnhancedQuery(
                original_query="test",
                strategy_used=EnhancementStrategy.EXPANSION,
                confidence_score=1.5,  # Out of range
            )

        with pytest.raises(ValueError):
            EnhancedQuery(
                original_query="test",
                strategy_used=EnhancementStrategy.EXPANSION,
                confidence_score=-0.1,  # Out of range
            )

    def test_enhanced_query_metadata(self) -> None:
        """Test storing metadata in EnhancedQuery."""
        query = EnhancedQuery(
            original_query="test query",
            strategy_used=EnhancementStrategy.EXPANSION,
            metadata={"source": "llm", "model": "gpt-3.5-turbo"},
        )

        assert query.metadata["source"] == "llm"
        assert query.metadata["model"] == "gpt-3.5-turbo"


class TestQueryEnhancerConfig:
    """Test QueryEnhancerConfig configuration."""

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        config = QueryEnhancerConfig()

        assert config.strategy == EnhancementStrategy.EXPANSION
        assert config.max_expansions == 5
        assert config.min_expansion_length == 2
        assert config.include_original is True
        assert config.normalize_query is True
        assert config.cache_enabled is True
        assert config.timeout_seconds == 5.0

    def test_config_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = QueryEnhancerConfig(
            strategy=EnhancementStrategy.HYBRID,
            max_expansions=10,
            timeout_seconds=10.0,
            cache_enabled=False,
        )

        assert config.strategy == EnhancementStrategy.HYBRID
        assert config.max_expansions == 10
        assert config.timeout_seconds == 10.0
        assert config.cache_enabled is False

    def test_config_validation_max_expansions(self) -> None:
        """Test max_expansions validation."""
        with pytest.raises(ValueError):
            QueryEnhancerConfig(max_expansions=0)  # Must be >= 1

    def test_config_validation_timeout(self) -> None:
        """Test timeout validation."""
        with pytest.raises(ValueError):
            QueryEnhancerConfig(timeout_seconds=0)  # Must be > 0


class TestLLMQueryEnhancerConfig:
    """Test LLMQueryEnhancerConfig configuration."""

    def test_llm_config_defaults(self) -> None:
        """Test default LLM configuration."""
        config = LLMQueryEnhancerConfig()

        assert config.llm_provider == "openai"
        assert config.model_name == "gpt-3.5-turbo"
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.max_tokens == 256
        assert config.enable_caching is True
        assert config.batch_size == 5

    def test_llm_config_custom(self) -> None:
        """Test creating LLM config with custom values."""
        config = LLMQueryEnhancerConfig(
            llm_provider="anthropic", model_name="claude-3-sonnet", temperature=0.5, batch_size=10
        )

        assert config.llm_provider == "anthropic"
        assert config.model_name == "claude-3-sonnet"
        assert config.temperature == 0.5
        assert config.batch_size == 10

    def test_llm_config_temperature_validation(self) -> None:
        """Test temperature parameter validation."""
        with pytest.raises(ValueError):
            LLMQueryEnhancerConfig(temperature=-0.1)

        with pytest.raises(ValueError):
            LLMQueryEnhancerConfig(temperature=2.5)

    def test_llm_config_top_p_validation(self) -> None:
        """Test top_p parameter validation."""
        with pytest.raises(ValueError):
            LLMQueryEnhancerConfig(top_p=-0.1)

        with pytest.raises(ValueError):
            LLMQueryEnhancerConfig(top_p=1.1)


@pytest.mark.asyncio
class TestLLMQueryEnhancer:
    """Test LLM query enhancer functionality."""

    async def test_initialization(self) -> None:
        """Test initializing LLM query enhancer."""
        config = LLMQueryEnhancerConfig()
        enhancer = LLMQueryEnhancer(config)

        assert enhancer.config == config
        assert enhancer._cache == {}
        assert enhancer._llm_client is None

    async def test_initialization_with_defaults(self) -> None:
        """Test initialization with default config."""
        enhancer = LLMQueryEnhancer()

        assert enhancer.config is not None
        assert isinstance(enhancer.config, LLMQueryEnhancerConfig)
        assert enhancer.config.model_name == "gpt-3.5-turbo"

    async def test_cache_key_generation(self) -> None:
        """Test generating cache keys."""
        enhancer = LLMQueryEnhancer()

        key1 = enhancer._get_cache_key("Test Query", "expand")
        key2 = enhancer._get_cache_key("test query", "expand")

        # Keys should be case-insensitive
        assert key1 == key2
        assert "expand" in key1
        assert "test query" in key1.lower()

    async def test_cache_operations(self) -> None:
        """Test cache get/put operations."""
        enhancer = LLMQueryEnhancer()

        key = "expand:test query"
        value = ["result1", "result2"]

        # Cache empty initially
        assert enhancer._get_cached_result(key) is None

        # Store in cache
        enhancer._cache_result(key, value)

        # Retrieve from cache
        cached = enhancer._get_cached_result(key)
        assert cached == value

    async def test_cache_disabled(self) -> None:
        """Test caching can be disabled."""
        config = LLMQueryEnhancerConfig(cache_enabled=False)
        enhancer = LLMQueryEnhancer(config)

        key = "test_key"
        enhancer._cache_result(key, "value")

        # Should return None even if key was added
        cached = enhancer._get_cached_result(key)
        assert cached is None

    async def test_validate_query_empty(self) -> None:
        """Test query validation rejects empty queries."""
        enhancer = LLMQueryEnhancer()

        with pytest.raises(ValueError):
            enhancer._validate_query("")

        with pytest.raises(ValueError):
            enhancer._validate_query("   ")

    async def test_validate_query_too_long(self) -> None:
        """Test query validation rejects very long queries."""
        enhancer = LLMQueryEnhancer()

        long_query = "x" * 10001
        with pytest.raises(ValueError):
            enhancer._validate_query(long_query)

    async def test_normalize_query(self) -> None:
        """Test query normalization."""
        enhancer = LLMQueryEnhancer()

        normalized = enhancer._normalize_query("  Test Query  ")
        assert normalized == "test query"

    async def test_normalize_query_disabled(self) -> None:
        """Test query normalization can be disabled."""
        config = LLMQueryEnhancerConfig(normalize_query=False)
        enhancer = LLMQueryEnhancer(config)

        result = enhancer._normalize_query("  Test Query  ")
        assert result == "  Test Query  "

    async def test_parse_expansion_response(self) -> None:
        """Test parsing expansion response from LLM."""
        enhancer = LLMQueryEnhancer()

        response = "1. retrieval augmented generation\n2. rag\n3. document retrieval"
        terms = enhancer._parse_expansion_response(response)

        assert len(terms) >= 2
        assert "retrieval augmented generation" in terms or "rag" in terms

    async def test_parse_expansion_response_with_bullets(self) -> None:
        """Test parsing expansion response with bullet points."""
        enhancer = LLMQueryEnhancer()

        response = "- term1\n• term2\n* term3"
        terms = enhancer._parse_expansion_response(response)

        assert len(terms) == 3

    async def test_parse_rewrite_response(self) -> None:
        """Test parsing rewrite response from LLM."""
        enhancer = LLMQueryEnhancer()

        response = "  What are machine learning models?  "
        rewritten = enhancer._parse_rewrite_response(response)

        assert rewritten == "What are machine learning models?"


@pytest.mark.asyncio
class TestQueryExpansion:
    """Test query expansion functionality."""

    async def test_expand_query_with_mock_llm(self) -> None:
        """Test expanding query with mocked LLM."""
        config = LLMQueryEnhancerConfig()
        enhancer = LLMQueryEnhancer(config)

        # Mock LLM response
        mock_response = "1. retrieval augmented generation\n2. rag\n3. document retrieval"

        with patch.object(enhancer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await enhancer.expand_query("What is RAG?")

            assert len(result) > 0
            assert isinstance(result, list)
            mock_llm.assert_called_once()

    async def test_expand_query_includes_original(self) -> None:
        """Test expansion includes original query when configured."""
        config = LLMQueryEnhancerConfig(include_original=True)
        enhancer = LLMQueryEnhancer(config)

        mock_response = "1. expansion1\n2. expansion2"

        with patch.object(enhancer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await enhancer.expand_query("test query")

            # Should include original
            assert "test query" in result

    async def test_expand_query_excludes_original(self) -> None:
        """Test expansion can exclude original query."""
        config = LLMQueryEnhancerConfig(include_original=False)
        enhancer = LLMQueryEnhancer(config)

        mock_response = "1. expansion1\n2. expansion2"

        with patch.object(enhancer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await enhancer.expand_query("test query")

            # Should not include original
            assert "test query" not in result

    async def test_expand_query_respects_max_expansions(self) -> None:
        """Test expansion respects max_expansions limit."""
        config = LLMQueryEnhancerConfig(max_expansions=2, include_original=False)
        enhancer = LLMQueryEnhancer(config)

        mock_response = "1. exp1\n2. exp2\n3. exp3\n4. exp4\n5. exp5"

        with patch.object(enhancer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await enhancer.expand_query("test query")

            # Should respect limit
            assert len(result) <= 2

    async def test_expand_query_caches_result(self) -> None:
        """Test expansion results are cached."""
        config = LLMQueryEnhancerConfig(cache_enabled=True)
        enhancer = LLMQueryEnhancer(config)

        mock_response = "1. result1\n2. result2"

        with patch.object(enhancer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            # First call
            result1 = await enhancer.expand_query("test query")

            # Second call should use cache
            result2 = await enhancer.expand_query("test query")

            # LLM should only be called once
            assert mock_llm.call_count == 1
            assert result1 == result2

    async def test_expand_query_fallback_on_error(self) -> None:
        """Test expansion falls back to original on error."""
        enhancer = LLMQueryEnhancer()

        with patch.object(enhancer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("LLM error")

            result = await enhancer.expand_query("test query")

            # Should return normalized original
            assert "test query" in result


@pytest.mark.asyncio
class TestQueryRewriting:
    """Test query rewriting functionality."""

    async def test_rewrite_query_with_mock_llm(self) -> None:
        """Test rewriting query with mocked LLM."""
        enhancer = LLMQueryEnhancer()

        mock_response = "What are machine learning models?"

        with patch.object(enhancer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await enhancer.rewrite_query("ml models?")

            assert result == "What are machine learning models?"
            mock_llm.assert_called_once()

    async def test_rewrite_query_caches_result(self) -> None:
        """Test rewrite results are cached."""
        config = LLMQueryEnhancerConfig(cache_enabled=True)
        enhancer = LLMQueryEnhancer(config)

        mock_response = "What is retrieval augmented generation?"

        with patch.object(enhancer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            # First call
            result1 = await enhancer.rewrite_query("What is RAG?")

            # Second call should use cache
            result2 = await enhancer.rewrite_query("What is RAG?")

            # LLM should only be called once
            assert mock_llm.call_count == 1
            assert result1 == result2

    async def test_rewrite_query_fallback_on_error(self) -> None:
        """Test rewriting falls back to normalized original on error."""
        enhancer = LLMQueryEnhancer()

        with patch.object(enhancer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("LLM error")

            result = await enhancer.rewrite_query("test query")

            # Should return normalized original
            assert "test" in result.lower()


@pytest.mark.asyncio
class TestEnhancementStrategies:
    """Test different enhancement strategies."""

    async def test_enhance_expansion_strategy(self) -> None:
        """Test enhancement with expansion strategy."""
        enhancer = LLMQueryEnhancer()

        with patch.object(enhancer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "1. rag\n2. document retrieval"

            result = await enhancer.enhance("What is RAG?", strategy=EnhancementStrategy.EXPANSION)

            assert result.strategy_used == EnhancementStrategy.EXPANSION
            assert len(result.expanded_queries) > 0
            assert result.original_query == "What is RAG?"

    async def test_enhance_rewrite_strategy(self) -> None:
        """Test enhancement with rewrite strategy."""
        enhancer = LLMQueryEnhancer()

        with patch.object(enhancer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "What is retrieval augmented generation?"

            result = await enhancer.enhance("What is RAG?", strategy=EnhancementStrategy.REWRITE)

            assert result.strategy_used == EnhancementStrategy.REWRITE
            assert result.rewritten_query is not None
            assert result.original_query == "What is RAG?"

    async def test_enhance_hybrid_strategy(self) -> None:
        """Test enhancement with hybrid strategy."""
        enhancer = LLMQueryEnhancer()

        with patch.object(enhancer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            # First call returns expansions, second returns rewrite
            mock_llm.side_effect = [
                "1. rag\n2. retrieval augmented generation",
                "What is retrieval augmented generation?",
            ]

            result = await enhancer.enhance("What is RAG?", strategy=EnhancementStrategy.HYBRID)

            assert result.strategy_used == EnhancementStrategy.HYBRID
            assert len(result.expanded_queries) > 0
            assert result.rewritten_query is not None

    async def test_enhance_uses_default_strategy(self) -> None:
        """Test enhancement uses config default strategy."""
        config = LLMQueryEnhancerConfig(strategy=EnhancementStrategy.REWRITE)
        enhancer = LLMQueryEnhancer(config)

        with patch.object(enhancer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Rewritten query"

            result = await enhancer.enhance("original")

            # Should use rewrite strategy from config
            assert result.strategy_used == EnhancementStrategy.REWRITE


@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and error handling."""

    async def test_enhance_empty_query_raises_error(self) -> None:
        """Test enhancement raises error on empty query."""
        enhancer = LLMQueryEnhancer()

        with pytest.raises(ValueError):
            await enhancer.enhance("")

    async def test_enhance_very_long_query_raises_error(self) -> None:
        """Test enhancement raises error on very long query."""
        enhancer = LLMQueryEnhancer()

        long_query = "x" * 10001
        with pytest.raises(ValueError):
            await enhancer.enhance(long_query)

    async def test_enhance_special_characters(self) -> None:
        """Test enhancement handles special characters."""
        enhancer = LLMQueryEnhancer()

        with patch.object(enhancer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "result"

            query = "What is @#$%? & <query>?"
            result = await enhancer.enhance(query)

            assert result.original_query == query

    async def test_batch_enhance_empty_list(self) -> None:
        """Test batch enhancement with empty list."""
        enhancer = LLMQueryEnhancer()

        result = await enhancer.batch_enhance([])

        assert result == []

    async def test_batch_enhance_multiple_queries(self) -> None:
        """Test batch enhancement with multiple queries."""
        enhancer = LLMQueryEnhancer()

        with patch.object(enhancer, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "result"

            queries = ["query1", "query2", "query3"]
            results = await enhancer.batch_enhance(queries)

            assert len(results) == 3
            assert all(isinstance(r, EnhancedQuery) for r in results)


@pytest.mark.asyncio
class TestSingletonPattern:
    """Test singleton pattern for query enhancer."""

    async def test_get_enhancer_creates_instance(self) -> None:
        """Test get_llm_query_enhancer creates instance."""
        reset_llm_query_enhancer_instance()

        enhancer = get_llm_query_enhancer()

        assert enhancer is not None
        assert isinstance(enhancer, LLMQueryEnhancer)

    async def test_get_enhancer_returns_same_instance(self) -> None:
        """Test get_llm_query_enhancer returns same instance."""
        reset_llm_query_enhancer_instance()

        enhancer1 = get_llm_query_enhancer()
        enhancer2 = get_llm_query_enhancer()

        assert enhancer1 is enhancer2

    async def test_reset_enhancer_instance(self) -> None:
        """Test resetting enhancer singleton."""
        enhancer1 = get_llm_query_enhancer()
        reset_llm_query_enhancer_instance()
        enhancer2 = get_llm_query_enhancer()

        # Should be different instances
        assert enhancer1 is not enhancer2

    async def test_get_enhancer_with_custom_config(self) -> None:
        """Test creating singleton with custom config."""
        reset_llm_query_enhancer_instance()

        config = LLMQueryEnhancerConfig(model_name="custom-model", temperature=0.5)
        enhancer = get_llm_query_enhancer(config)

        assert enhancer.config.model_name == "custom-model"
        assert enhancer.config.temperature == 0.5
