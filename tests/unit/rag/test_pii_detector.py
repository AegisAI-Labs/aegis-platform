"""Unit tests for PII Detection and Masking module.

Tests cover:
- PII detection for 10+ types
- Confidence threshold filtering
- Text masking with various configurations
- Dictionary masking
- Error handling and graceful degradation
- Async wrappers
"""

from unittest.mock import MagicMock, patch

import pytest

from aegis_platform.rag.pii_detector import PII, PIIConfig, PIIDetector, get_pii_detector


class TestPIIConfig:
    """Test PIIConfig dataclass."""

    def test_default_config(self):
        """Test default PIIConfig values."""
        config = PIIConfig()
        assert config.enabled is True
        assert config.mask_char == "[REDACTED]"
        assert config.confidence_threshold == 0.4
        assert config.detect_names is False
        assert config.detect_addresses is False

    def test_custom_config(self):
        """Test custom PIIConfig values."""
        config = PIIConfig(
            enabled=False,
            mask_char="***",
            confidence_threshold=0.85,
            detect_names=True,
            detect_addresses=True,
        )
        assert config.enabled is False
        assert config.mask_char == "***"
        assert config.confidence_threshold == 0.85
        assert config.detect_names is True
        assert config.detect_addresses is True


class TestPII:
    """Test PII entity class."""

    def test_pii_creation(self):
        """Test PII entity creation."""
        pii = PII(
            entity_type="EMAIL_ADDRESS",
            text="john@example.com",
            start=0,
            end=16,
            confidence=0.95,
        )
        assert pii.entity_type == "EMAIL_ADDRESS"
        assert pii.text == "john@example.com"
        assert pii.start == 0
        assert pii.end == 16
        assert pii.confidence == 0.95

    def test_pii_repr(self):
        """Test PII string representation."""
        pii = PII("US_SSN", "123-45-6789", 0, 11, 0.98)
        repr_str = repr(pii)
        assert "US_SSN" in repr_str
        assert "123-45-6789" in repr_str or "123-45-67" in repr_str
        assert "0.98" in repr_str


class TestPIIDetectorInit:
    """Test PIIDetector initialization."""

    def test_init_success(self, mock_presidio):
        """Test successful PIIDetector initialization."""
        config = PIIConfig()
        detector = PIIDetector(config)

        assert detector.config == config
        assert detector.analyzer is not None
        assert detector.anonymizer is not None

    def test_init_missing_presidio(self):
        """Test initialization when Presidio is not installed."""
        with (
            patch("aegis_platform.rag.pii_detector.AnalyzerEngine", None),
            patch("aegis_platform.rag.pii_detector.AnonymizerEngine", None),
            pytest.raises(ImportError, match="Presidio library required"),
        ):
            PIIDetector()

    def test_init_with_default_config(self, mock_presidio):
        """Test initialization with default config."""
        detector = PIIDetector()
        assert detector.config.enabled is True
        assert detector.config.mask_char == "[REDACTED]"


class TestGetSupportedEntities:
    """Test get_supported_entities method."""

    def test_core_entities_only(self, mock_presidio):
        """Test supported entities without optional types."""
        config = PIIConfig(detect_names=False, detect_addresses=False)
        detector = PIIDetector(config)

        entities = detector.get_supported_entities()

        # Should include all core types
        for entity in detector.CORE_PII_TYPES:
            assert entity in entities

        # Should not include optional types
        assert "PERSON" not in entities
        assert "LOCATION" not in entities

    def test_with_optional_entities(self, mock_presidio):
        """Test supported entities with optional types enabled."""
        config = PIIConfig(detect_names=True, detect_addresses=True)
        detector = PIIDetector(config)

        entities = detector.get_supported_entities()

        # Should include all core types
        for entity in detector.CORE_PII_TYPES:
            assert entity in entities

        # Should include optional types
        assert "PERSON" in entities
        assert "LOCATION" in entities


class TestPIIDetection:
    """Test PII detection functionality."""

    def test_detect_pii_empty_text(self, mock_presidio):
        """Test detection with empty text."""
        detector = PIIDetector()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            detector.detect_pii("")

    def test_detect_pii_no_matches(self, mock_presidio):
        """Test detection when no PII is found."""
        detector = PIIDetector()
        detector.analyzer.analyze.return_value = []

        results = detector.detect_pii("This is a normal text with no PII.")

        assert len(results) == 0

    def test_detect_pii_with_matches(self, mock_presidio):
        """Test detection with PII matches."""
        detector = PIIDetector(PIIConfig(confidence_threshold=0.7))

        # Mock Presidio analyzer results
        mock_result = MagicMock()
        mock_result.entity_type = "EMAIL_ADDRESS"
        mock_result.start = 25
        mock_result.end = 42
        mock_result.score = 0.95

        detector.analyzer.analyze.return_value = [mock_result]

        text = "Contact me at john.doe@example.com for details."
        results = detector.detect_pii(text)

        assert len(results) == 1
        assert results[0].entity_type == "EMAIL_ADDRESS"
        assert results[0].confidence == 0.95

    def test_detect_pii_confidence_filtering(self, mock_presidio):
        """Test that low-confidence detections are filtered."""
        config = PIIConfig(confidence_threshold=0.80)
        detector = PIIDetector(config)

        # Create mock results with varying confidence
        high_conf = MagicMock()
        high_conf.entity_type = "EMAIL_ADDRESS"
        high_conf.start = 0
        high_conf.end = 16
        high_conf.score = 0.95

        low_conf = MagicMock()
        low_conf.entity_type = "PHONE_NUMBER"
        low_conf.start = 20
        low_conf.end = 32
        low_conf.score = 0.65

        detector.analyzer.analyze.return_value = [high_conf, low_conf]

        results = detector.detect_pii("Contact: john@example.com (555-123-4567)")

        # Only high confidence should be included
        assert len(results) == 1
        assert results[0].entity_type == "EMAIL_ADDRESS"

    def test_detect_pii_error_handling(self, mock_presidio):
        """Test error handling during detection."""
        detector = PIIDetector()
        detector.analyzer.analyze.side_effect = RuntimeError("Analyzer failed")

        with pytest.raises(RuntimeError, match="Analyzer failed"):
            detector.detect_pii("some text")


class TestPIIMasking:
    """Test PII masking functionality."""

    def test_mask_pii_empty_text(self, mock_presidio):
        """Test masking with empty text."""
        detector = PIIDetector()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            detector.mask_pii("")

    def test_mask_pii_no_pii(self, mock_presidio):
        """Test masking when no PII is detected."""
        detector = PIIDetector()
        detector.analyzer.analyze.return_value = []

        # Set up mock to return the input text unchanged
        detector.anonymizer.anonymize.return_value.text = "This is a normal text with no PII."

        text = "This is a normal text with no PII."
        result = detector.mask_pii(text)

        # When no PII is detected, text should be unchanged
        assert result == text

    def test_mask_pii_with_email(self, mock_presidio):
        """Test masking email addresses."""
        detector = PIIDetector()

        # Mock detection
        mock_result = MagicMock()
        mock_result.entity_type = "EMAIL_ADDRESS"
        mock_result.start = 25
        mock_result.end = 42
        mock_result.score = 0.95

        detector.analyzer.analyze.return_value = [mock_result]

        # Mock anonymization - return text with email replaced
        detector.anonymizer.anonymize.return_value.text = "Contact me at [REDACTED] for details."

        text = "Contact me at john.doe@example.com for details."
        result = detector.mask_pii(text)

        assert "[REDACTED]" in result

    def test_mask_pii_custom_mask_char(self, mock_presidio):
        """Test masking with custom mask character."""
        config = PIIConfig(mask_char="***")
        detector = PIIDetector(config)

        # Mock detection
        mock_result = MagicMock()
        mock_result.entity_type = "US_SSN"
        mock_result.start = 0
        mock_result.end = 11
        mock_result.score = 0.99

        detector.analyzer.analyze.return_value = [mock_result]
        detector.anonymizer.anonymize.return_value.text = "***"

        result = detector.mask_pii("123-45-6789")

        assert "***" in result

    def test_mask_pii_error_fails_closed(self, mock_presidio):
        """Masking errors raise instead of returning the unmasked text."""
        detector = PIIDetector()
        detector.anonymizer.anonymize.side_effect = RuntimeError("Anonymizer failed")

        # Mock detection so we have PII to mask
        mock_result = MagicMock()
        mock_result.entity_type = "EMAIL_ADDRESS"
        mock_result.start = 0
        mock_result.end = 16
        mock_result.score = 0.95

        detector.analyzer.analyze.return_value = [mock_result]

        with pytest.raises(RuntimeError, match="PII masking failed"):
            detector.mask_pii("john.doe@example.com")

    def test_mask_dict_error_fails_closed(self, mock_presidio):
        """Dictionary masking propagates errors instead of returning the input."""
        detector = PIIDetector()
        detector.anonymizer.anonymize.side_effect = RuntimeError("Anonymizer failed")

        with pytest.raises(RuntimeError, match="PII masking failed"):
            detector.mask_pii_in_dict({"email": "john.doe@example.com"})


class TestDictionaryMasking:
    """Test dictionary masking functionality."""

    def test_mask_dict_all_strings(self, mock_presidio):
        """Test masking all string values in a dictionary."""
        detector = PIIDetector()

        # Mock detection - return empty (no PII)
        detector.analyzer.analyze.return_value = []

        # Setup mock to return text unchanged
        detector.anonymizer.anonymize.return_value.text = "John Doe"

        data = {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "age": 30,
            "active": True,
        }

        # Pass keys_to_mask=None to mask all strings
        result = detector.mask_pii_in_dict(data, keys_to_mask=None)

        # Should process all string values (no PII detected, so unchanged)
        assert isinstance(result["name"], str)
        assert isinstance(result["email"], str)
        # Non-string values should remain unchanged
        assert result["age"] == 30
        assert result["active"] is True

    def test_mask_dict_specific_keys(self, mock_presidio):
        """Test masking specific keys in a dictionary."""
        detector = PIIDetector()

        # Mock detection
        mock_result = MagicMock()
        mock_result.entity_type = "EMAIL_ADDRESS"
        mock_result.start = 0
        mock_result.end = 16
        mock_result.score = 0.95

        detector.analyzer.analyze.return_value = [mock_result]
        detector.anonymizer.anonymize.return_value.text = "[REDACTED]"

        data = {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "555-123-4567",
        }

        result = detector.mask_pii_in_dict(data, keys_to_mask=["email"])

        # Only email should be processed
        assert result["name"] == "John Doe"
        assert result["phone"] == "555-123-4567"


class TestAsyncWrappers:
    """Test async wrapper methods."""

    @pytest.mark.asyncio
    async def test_async_detect_pii(self, mock_presidio):
        """Test async PII detection."""
        detector = PIIDetector()
        detector.analyzer.analyze.return_value = []

        result = await detector.async_detect_pii("test text")

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_async_mask_pii(self, mock_presidio):
        """Test async PII masking."""
        detector = PIIDetector()
        detector.analyzer.analyze.return_value = []
        detector.anonymizer.anonymize.return_value.text = "test text"

        result = await detector.async_mask_pii("test text")

        # Should return string
        assert isinstance(result, str)
        assert result == "test text"


class TestSingletonPattern:
    """Test singleton pattern for PIIDetector."""

    @pytest.fixture(autouse=True)
    def _reset_singleton(self):
        import aegis_platform.rag.pii_detector as pii_module

        yield
        pii_module._pii_detector = None

    def test_singleton_creation(self, mock_presidio):
        """Test that singleton is created correctly."""
        # Reset the global singleton
        import aegis_platform.rag.pii_detector as pii_module

        pii_module._pii_detector = None

        detector1 = get_pii_detector()
        detector2 = get_pii_detector()

        assert detector1 is detector2

    def test_singleton_config_first_call(self, mock_presidio):
        """Test that config from first call is used."""
        # Reset the global singleton
        import aegis_platform.rag.pii_detector as pii_module

        pii_module._pii_detector = None

        config = PIIConfig(mask_char="***")
        detector = get_pii_detector(config)

        assert detector.config.mask_char == "***"
