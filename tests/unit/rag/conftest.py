"""Pytest configuration for RAG unit tests."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock sentence_transformers before any test imports it
sys.modules["sentence_transformers"] = MagicMock()


@pytest.fixture
def mock_presidio():
    """Mock Presidio engines for testing."""
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

        # Configure anonymizer.anonymize() to return an object with a text attribute
        # This is important for mask_pii() which accesses .text on the result
        mock_anonymize_result = MagicMock()
        mock_anonymize_result.text = ""  # Will be overridden per test
        mock_anonymizer_instance.anonymize.return_value = mock_anonymize_result

        yield {
            "analyzer": mock_analyzer_class,
            "anonymizer": mock_anonymizer_class,
            "analyzer_instance": mock_analyzer_instance,
            "anonymizer_instance": mock_anonymizer_instance,
        }
