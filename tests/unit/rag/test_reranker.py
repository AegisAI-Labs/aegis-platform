"""Unit tests for reranking modules.

Tests coverage:
- Base reranker interface and models
- Cross-encoder reranker implementation
- Edge cases and error handling
- Singleton pattern
- Configuration validation
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from aegis_platform.rag.cross_encoder_reranker import (
    CrossEncoderConfig,
    CrossEncoderReranker,
    get_cross_encoder_reranker,
    reset_cross_encoder_instance,
)
from aegis_platform.rag.reranker_base import (
    RankedResult,
    RankerStrategy,
    RawResult,
    RerankerConfig,
)

sys.modules["sentence_transformers"] = MagicMock()


class TestRankedResult:
    """Tests for RankedResult model."""

    def test_ranked_result_creation(self) -> None:
        """Test creating a ranked result."""
        result = RankedResult(
            result_id="1",
            text="RAG improves LLM accuracy",
            relevance_score=0.95,
            source_id="doc_1",
            original_rank=1,
            new_rank=1,
        )

        assert result.result_id == "1"
        assert result.text == "RAG improves LLM accuracy"
        assert result.relevance_score == 0.95
        assert result.source_id == "doc_1"
        assert result.original_rank == 1
        assert result.new_rank == 1

    def test_ranked_result_score_validation(self) -> None:
        """Test that score must be 0-1 range."""
        with pytest.raises(ValueError):
            RankedResult(
                result_id="1", text="text", relevance_score=1.5, original_rank=1, new_rank=1
            )

    def test_ranked_result_metadata(self) -> None:
        """Test ranked result with metadata."""
        metadata = {"chunk_id": "c_1", "page": 5}

        result = RankedResult(
            result_id="1",
            text="text",
            relevance_score=0.8,
            metadata=metadata,
            original_rank=1,
            new_rank=1,
        )

        assert result.metadata == metadata


class TestRerankerConfig:
    """Tests for RerankerConfig model."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = RerankerConfig()

        assert config.strategy == RankerStrategy.DENSE
        assert config.batch_size == 32
        assert config.score_threshold == 0.0
        assert config.top_k == 10
        assert config.normalize_scores is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = RerankerConfig(
            strategy=RankerStrategy.HYBRID, batch_size=16, score_threshold=0.5, top_k=20
        )

        assert config.strategy == RankerStrategy.HYBRID
        assert config.batch_size == 16
        assert config.score_threshold == 0.5
        assert config.top_k == 20

    def test_config_validation(self) -> None:
        """Test configuration validation."""
        with pytest.raises(ValueError):
            RerankerConfig(batch_size=0)

        with pytest.raises(ValueError):
            RerankerConfig(score_threshold=1.5)

        with pytest.raises(ValueError):
            RerankerConfig(top_k=-1)


class TestCrossEncoderConfig:
    """Tests for CrossEncoderConfig model."""

    def test_cross_encoder_config_defaults(self) -> None:
        """Test cross-encoder config defaults."""
        config = CrossEncoderConfig()

        assert config.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert config.device == "cpu"
        assert config.max_length == 512
        assert config.strategy == RankerStrategy.DENSE

    def test_cross_encoder_config_custom(self) -> None:
        """Test custom cross-encoder config."""
        config = CrossEncoderConfig(
            model_name="cross-encoder/qnli-distilroberta-base", device="cuda", batch_size=64
        )

        assert config.model_name == "cross-encoder/qnli-distilroberta-base"
        assert config.device == "cuda"
        assert config.batch_size == 64


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker implementation."""

    @pytest.fixture
    def config(self) -> CrossEncoderConfig:
        """Create test configuration."""
        return CrossEncoderConfig()

    @pytest.fixture
    def reranker(self, config: CrossEncoderConfig) -> CrossEncoderReranker:
        """Create test reranker."""
        return CrossEncoderReranker(config)

    @pytest.fixture
    def raw_results(self) -> list[RawResult]:
        """Create test raw results."""
        return [
            RawResult(
                result_id="1",
                text="RAG combines retrieval and generation",
                similarity_score=0.9,
                source_id="doc_1",
            ),
            RawResult(
                result_id="2",
                text="Vector search finds similar documents",
                similarity_score=0.85,
                source_id="doc_2",
            ),
            RawResult(
                result_id="3",
                text="Machine learning models improve over time",
                similarity_score=0.7,
                source_id="doc_3",
            ),
        ]

    def test_reranker_initialization(self) -> None:
        """Test reranker initialization."""
        config = CrossEncoderConfig()
        reranker = CrossEncoderReranker(config)

        assert reranker.config == config
        assert reranker.model is None
        assert reranker._model_loaded is False

    @patch("sentence_transformers.CrossEncoder")
    @pytest.mark.asyncio
    async def test_score_query_document_pair(
        self, mock_cross_encoder_class: MagicMock, reranker: CrossEncoderReranker
    ) -> None:
        """Test scoring a query-document pair."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.85]
        mock_cross_encoder_class.return_value = mock_model

        query = "What is RAG?"
        document = "RAG combines retrieval and generation"

        score = await reranker.score(query, document)

        assert 0.0 <= score <= 1.0
        assert mock_model.predict.called

    @patch("sentence_transformers.CrossEncoder")
    @pytest.mark.asyncio
    async def test_batch_score_documents(
        self, mock_cross_encoder_class: MagicMock, reranker: CrossEncoderReranker
    ) -> None:
        """Test batch scoring multiple documents."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.85, 0.7]
        mock_cross_encoder_class.return_value = mock_model

        query = "What is RAG?"
        documents = [
            "RAG combines retrieval and generation",
            "Vector search finds similar documents",
            "Machine learning improves over time",
        ]

        scores = await reranker.batch_score(query, documents)

        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert mock_model.predict.called

    @patch("sentence_transformers.CrossEncoder")
    @pytest.mark.asyncio
    async def test_rerank_results(
        self,
        mock_cross_encoder_class: MagicMock,
        reranker: CrossEncoderReranker,
        raw_results: list[RawResult],
    ) -> None:
        """Test reranking search results."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.95, 0.75, 0.6]
        mock_cross_encoder_class.return_value = mock_model

        query = "What is RAG?"

        reranked = await reranker.rerank(query, raw_results, top_k=2)

        assert len(reranked) == 2
        assert reranked[0].relevance_score >= reranked[1].relevance_score
        assert all(0.0 <= r.relevance_score <= 1.0 for r in reranked)

    @pytest.mark.asyncio
    async def test_rerank_empty_results_raises(self, reranker: CrossEncoderReranker) -> None:
        """Test that empty results raises error."""
        with pytest.raises(ValueError):
            await reranker.rerank("query", [])

    @pytest.mark.asyncio
    async def test_rerank_empty_query_raises(
        self, reranker: CrossEncoderReranker, raw_results: list[RawResult]
    ) -> None:
        """Test that empty query raises error."""
        with pytest.raises(ValueError):
            await reranker.rerank("", raw_results)

    @patch("sentence_transformers.CrossEncoder")
    @pytest.mark.asyncio
    async def test_rerank_preserves_metadata(
        self, mock_cross_encoder_class: MagicMock, reranker: CrossEncoderReranker
    ) -> None:
        """Test that reranking preserves result metadata."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8]
        mock_cross_encoder_class.return_value = mock_model

        query = "What is RAG?"
        results = [
            RawResult(
                result_id="1", text="RAG text", metadata={"chunk_id": "c_1"}, source_id="doc_1"
            ),
            RawResult(
                result_id="2",
                text="Text about retrieval",
                metadata={"chunk_id": "c_2"},
                source_id="doc_2",
            ),
        ]

        reranked = await reranker.rerank(query, results)

        assert len(reranked) == 2
        assert reranked[0].metadata["chunk_id"] == "c_1"
        assert reranked[1].metadata["chunk_id"] == "c_2"

    @patch("sentence_transformers.CrossEncoder")
    @pytest.mark.asyncio
    async def test_rerank_respects_threshold(self, mock_cross_encoder_class: MagicMock) -> None:
        """Test that reranking respects score threshold."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.5, 0.2]
        mock_cross_encoder_class.return_value = mock_model

        config = CrossEncoderConfig(score_threshold=0.6)
        reranker = CrossEncoderReranker(config)

        query = "What is RAG?"
        results = [
            RawResult(result_id="1", text="Good match"),
            RawResult(result_id="2", text="Medium match"),
            RawResult(result_id="3", text="Poor match"),
        ]

        reranked = await reranker.rerank(query, results)

        assert len(reranked) == 1
        assert reranked[0].relevance_score >= 0.6

    @patch("sentence_transformers.CrossEncoder")
    @pytest.mark.asyncio
    async def test_rerank_respects_top_k(self, mock_cross_encoder_class: MagicMock) -> None:
        """Test that reranking respects top-k limit."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45]
        mock_cross_encoder_class.return_value = mock_model

        reranker = CrossEncoderReranker()

        query = "What is RAG?"
        results = [RawResult(result_id=str(i), text=f"Result {i}") for i in range(10)]

        reranked = await reranker.rerank(query, results, top_k=3)

        assert len(reranked) == 3

    @patch("sentence_transformers.CrossEncoder")
    @pytest.mark.asyncio
    async def test_rerank_updates_ranks(
        self,
        mock_cross_encoder_class: MagicMock,
        reranker: CrossEncoderReranker,
        raw_results: list[RawResult],
    ) -> None:
        """Test that reranking updates rank fields."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.95, 0.75, 0.6]
        mock_cross_encoder_class.return_value = mock_model

        query = "What is RAG?"

        reranked = await reranker.rerank(query, raw_results)

        for i, result in enumerate(reranked, start=1):
            assert result.new_rank == i


class TestCrossEncoderScoring:
    """Tests for cross-encoder scoring methods."""

    @patch("sentence_transformers.CrossEncoder")
    @pytest.mark.asyncio
    async def test_score_validates_query(self, mock_cross_encoder_class: MagicMock) -> None:
        """Test that score validates query input."""
        reranker = CrossEncoderReranker()

        with pytest.raises(ValueError):
            await reranker.score("", "document")

        with pytest.raises(ValueError):
            await reranker.score("   ", "document")

    @patch("sentence_transformers.CrossEncoder")
    @pytest.mark.asyncio
    async def test_score_validates_document(self, mock_cross_encoder_class: MagicMock) -> None:
        """Test that score validates document input."""
        reranker = CrossEncoderReranker()

        with pytest.raises(ValueError):
            await reranker.score("query", "")

        with pytest.raises(ValueError):
            await reranker.score("query", "   ")

    @patch("sentence_transformers.CrossEncoder")
    @pytest.mark.asyncio
    async def test_batch_score_empty_documents(self, mock_cross_encoder_class: MagicMock) -> None:
        """Test batch score with empty documents list."""
        reranker = CrossEncoderReranker()

        scores = await reranker.batch_score("query", [])

        assert scores == []

    @patch("sentence_transformers.CrossEncoder")
    @pytest.mark.asyncio
    async def test_batch_score_validates_documents(
        self, mock_cross_encoder_class: MagicMock
    ) -> None:
        """Test batch score validates all documents."""
        reranker = CrossEncoderReranker()

        with pytest.raises(ValueError):
            await reranker.batch_score("query", ["doc1", "", "doc3"])


class TestCrossEncoderEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("sentence_transformers.CrossEncoder")
    @pytest.mark.asyncio
    async def test_very_long_query_and_document(self, mock_cross_encoder_class: MagicMock) -> None:
        """Test scoring with very long query and document."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]
        mock_cross_encoder_class.return_value = mock_model

        reranker = CrossEncoderReranker()

        long_query = "query " * 1000
        long_document = "document " * 1000

        score = await reranker.score(long_query, long_document)

        assert 0.0 <= score <= 1.0

    @patch("sentence_transformers.CrossEncoder")
    @pytest.mark.asyncio
    async def test_special_characters_in_text(self, mock_cross_encoder_class: MagicMock) -> None:
        """Test scoring with special characters."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.7]
        mock_cross_encoder_class.return_value = mock_model

        reranker = CrossEncoderReranker()

        query = "What is <RAG> & 'retrieval'?"
        document = "RAG (Retrieval-Augmented Generation) == RAG"

        score = await reranker.score(query, document)

        assert 0.0 <= score <= 1.0

    @patch("sentence_transformers.CrossEncoder")
    @pytest.mark.asyncio
    async def test_single_result_reranking(self, mock_cross_encoder_class: MagicMock) -> None:
        """Test reranking with single result."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]
        mock_cross_encoder_class.return_value = mock_model

        reranker = CrossEncoderReranker()

        query = "What is RAG?"
        results = [RawResult(result_id="1", text="RAG text")]

        reranked = await reranker.rerank(query, results)

        assert len(reranked) == 1
        assert reranked[0].new_rank == 1


class TestCrossEncoderSingletonPattern:
    """Tests for singleton pattern."""

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_cross_encoder_instance()

    def test_singleton_creation(self) -> None:
        """Test singleton instance creation."""
        reranker1 = get_cross_encoder_reranker()
        reranker2 = get_cross_encoder_reranker()

        assert reranker1 is reranker2

    def test_singleton_reset(self) -> None:
        """Test resetting singleton instance."""
        reranker1 = get_cross_encoder_reranker()

        reset_cross_encoder_instance()

        reranker2 = get_cross_encoder_reranker()

        assert reranker1 is not reranker2
