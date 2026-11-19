"""Pytest configuration for integration tests."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock sentence_transformers before any test imports it
sys.modules["sentence_transformers"] = MagicMock()


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for integration tests."""
    mock_client = AsyncMock()

    # Mock OpenAI response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[
        0
    ].message.content = "Query expansion: security practices, data protection, encryption methods"

    async def mock_create(*args, **kwargs):
        return mock_response

    mock_client.chat.completions.create = mock_create

    with patch("aegis_platform.rag.llm_query_enhancer.AsyncOpenAI", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_embeddings_provider():
    """Mock embeddings provider for integration tests."""
    mock_client = AsyncMock()

    # Mock embedding response
    async def mock_embed(*args, **kwargs):
        return [0.1, 0.2, 0.3, 0.4, 0.5] * 24  # 120-dim embedding

    mock_client.embed = mock_embed

    with patch("aegis_platform.rag.llm_query_enhancer.AsyncOpenAI"):
        yield mock_client


@pytest.fixture(autouse=True)
def mock_presidio():
    """Mock Presidio engines for integration tests."""
    with (
        patch("aegis_platform.rag.pii_detector.AnalyzerEngine") as mock_analyzer_class,
        patch("aegis_platform.rag.pii_detector.AnonymizerEngine") as mock_anonymizer_class,
    ):
        # Create instances
        mock_analyzer_instance = MagicMock()
        mock_anonymizer_instance = MagicMock()

        # Configure classes to return instances
        mock_analyzer_class.return_value = mock_analyzer_instance
        mock_anonymizer_class.return_value = mock_anonymizer_instance

        # Configure analyzer.analyze() to return PII detection results
        mock_analyzer_instance.analyze.return_value = [
            MagicMock(entity_type="CREDIT_CARD"),
            MagicMock(entity_type="SSN"),
        ]

        # Configure anonymizer.anonymize() to return an object with a text attribute
        mock_anonymize_result = MagicMock()
        mock_anonymize_result.text = "Call me at [PHONE_NUMBER] for support"
        mock_anonymizer_instance.anonymize.return_value = mock_anonymize_result

        yield {
            "analyzer": mock_analyzer_class,
            "anonymizer": mock_anonymizer_class,
            "analyzer_instance": mock_analyzer_instance,
            "anonymizer_instance": mock_anonymizer_instance,
        }
