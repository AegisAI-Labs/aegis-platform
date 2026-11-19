"""PII Detection and Masking for RAG Platform.

This module provides PII (Personally Identifiable Information) detection
and masking using the Presidio library. It detects and masks 10+ types of
sensitive data including:
- Email addresses
- Phone numbers (US format)
- Social Security Numbers (SSN)
- Credit card numbers
- Bank account numbers
- Passport numbers
- Driver license numbers
- IP addresses
- Names (optional, configurable)
- Addresses (optional, configurable)

The detector can mask PII in documents before ingestion and in search
results before returning to users, ensuring sensitive data protection.
"""

import logging
from dataclasses import dataclass
from typing import ClassVar

logger = logging.getLogger(__name__)

# Import Presidio engines at module level for testability
try:
    from presidio_analyzer import AnalyzerEngine  # type: ignore
    from presidio_anonymizer import AnonymizerEngine  # type: ignore
    from presidio_anonymizer.entities import OperatorConfig  # type: ignore
except ImportError:
    # Engines will be None if Presidio is not installed; __init__ will raise
    AnalyzerEngine = None  # type: ignore
    AnonymizerEngine = None  # type: ignore
    OperatorConfig = None  # type: ignore


@dataclass
class PIIConfig:
    """Configuration for PII Detection and Masking.

    Attributes:
        enabled: Whether PII detection is enabled
        mask_char: Character/string to use for masking (default: "[REDACTED]")
        confidence_threshold: Minimum Presidio score (0.0-1.0) for an entity to be
            detected or masked. Presidio scores pattern-only matches such as phone
            numbers at 0.4, so higher values silently skip them.
        detect_names: Whether to detect names (may cause false positives)
        detect_addresses: Whether to detect addresses (may cause false positives)
    """

    enabled: bool = True
    mask_char: str = "[REDACTED]"
    confidence_threshold: float = 0.4
    detect_names: bool = False
    detect_addresses: bool = False


class PII:
    """Represents a detected PII entity.

    Attributes:
        entity_type: Type of PII (EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, etc.)
        text: The detected PII text
        start: Start position in original text
        end: End position in original text
        confidence: Confidence score (0.0-1.0)
    """

    def __init__(
        self,
        entity_type: str,
        text: str,
        start: int,
        end: int,
        confidence: float = 1.0,
    ):
        self.entity_type = entity_type
        self.text = text
        self.start = start
        self.end = end
        self.confidence = confidence

    def __repr__(self) -> str:
        return (
            f"PII(type={self.entity_type}, text='{self.text[:20] if len(self.text) > 20 else self.text}', "
            f"pos={self.start}-{self.end}, conf={self.confidence:.2f})"
        )


class PIIDetector:
    """PII Detection and Masking using Presidio.

    This detector uses the Presidio library to identify and mask sensitive
    personal information in text. It supports 10+ PII types and can be
    configured to include/exclude optional detections.

    Supported PII types:
    - EMAIL_ADDRESS: Email addresses (name@domain.com)
    - PHONE_NUMBER: Phone numbers (US format: +1-555-123-4567, etc.)
    - US_SSN: Social Security Numbers (XXX-XX-XXXX)
    - CREDIT_CARD: Credit card numbers (Visa, Mastercard, Amex, Discover)
    - US_BANK_ACCOUNT: US bank account numbers
    - US_PASSPORT: US passport numbers
    - US_DRIVER_LICENSE: US driver license numbers
    - IP_ADDRESS: IPv4 and IPv6 addresses
    - PERSON: Names (optional, may have false positives)
    - LOCATION: Addresses (optional, may have false positives)
    """

    # PII entity types supported by Presidio
    CORE_PII_TYPES: ClassVar[set[str]] = {
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "US_BANK_ACCOUNT",
        "US_PASSPORT",
        "US_DRIVER_LICENSE",
        "IP_ADDRESS",
    }

    OPTIONAL_PII_TYPES: ClassVar[set[str]] = {
        "PERSON",
        "LOCATION",
    }

    def __init__(self, config: PIIConfig | None = None):
        """Initialize PII Detector with configuration.

        Args:
            config: PIIConfig object (uses defaults if None)

        Raises:
            ImportError: If Presidio is not installed
        """
        self.config = config or PIIConfig()

        if AnalyzerEngine is None or AnonymizerEngine is None:
            raise ImportError(
                "Presidio library required for PII detection. "
                "Install with: pip install presidio-analyzer presidio-anonymizer"
            )

        try:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            logger.info("PIIDetector initialized with Presidio")
        except Exception as e:
            logger.error("Failed to initialize Presidio: %s", e)
            raise

    def get_supported_entities(self) -> list[str]:
        """Get list of supported PII entity types.

        Returns:
            List of supported entity type strings
        """
        entities = list(self.CORE_PII_TYPES)
        if self.config.detect_names:
            entities.append("PERSON")
        if self.config.detect_addresses:
            entities.append("LOCATION")
        return entities

    def detect_pii(self, text: str) -> list[PII]:
        """Detect PII entities in text.

        Args:
            text: Text to scan for PII

        Returns:
            List of detected PII entities (may be empty if no PII found)

        Raises:
            ValueError: If text is None or empty
            RuntimeError: If Presidio detection fails
        """
        if not text:
            raise ValueError("Text cannot be empty")

        try:
            entities = self.get_supported_entities()
            results = self.analyzer.analyze(
                text=text,
                entities=entities,
                language="en",
            )

            # Filter by confidence threshold
            pii_list = [
                PII(
                    entity_type=result.entity_type,
                    text=text[result.start : result.end],
                    start=result.start,
                    end=result.end,
                    confidence=result.score,
                )
                for result in results
                if result.score >= self.config.confidence_threshold
            ]

            if pii_list:
                logger.debug(
                    "Detected %d PII entities in text of length %d",
                    len(pii_list),
                    len(text),
                )
                for pii in pii_list:
                    logger.debug("  - %s", pii)

            return pii_list

        except RuntimeError:
            raise
        except Exception as e:
            logger.error("PII detection failed: %s", e)
            raise RuntimeError(f"PII detection error: {e}") from e

    def mask_pii(self, text: str) -> str:
        """Mask PII entities in text.

        Replaces detected PII with the configured mask character.

        Args:
            text: Text to mask

        Returns:
            Text with PII masked

        Raises:
            ValueError: If text is None or empty
            RuntimeError: If masking fails
        """
        if not text:
            raise ValueError("Text cannot be empty")

        try:
            entities = self.get_supported_entities()
            analyzer_results = [
                result
                for result in self.analyzer.analyze(text=text, entities=entities, language="en")
                if result.score >= self.config.confidence_threshold
            ]
            replace = OperatorConfig("replace", {"new_value": self.config.mask_char})
            anonymized = self.anonymizer.anonymize(
                text=text,
                # Presidio's analyzer and anonymizer declare separate RecognizerResult
                # classes; the anonymizer accepts analyzer results at runtime.
                analyzer_results=analyzer_results,  # type: ignore[arg-type]
                operators={"DEFAULT": replace},
            )

            masked_text = anonymized.text

            if masked_text != text:
                logger.debug(
                    "Masked PII in text (original=%d, masked=%d)",
                    len(text),
                    len(masked_text),
                )

            return masked_text

        except Exception as e:
            # Fail closed: returning the original text here would leak the PII.
            logger.error("PII masking failed: %s", e)
            raise RuntimeError(f"PII masking failed: {e}") from e

    def mask_pii_in_dict(self, data: dict, keys_to_mask: list[str] | None = None) -> dict:
        """Mask PII in dictionary values.

        Args:
            data: Dictionary with potential PII values
            keys_to_mask: List of keys whose values should be masked
                         (if None, masks all string values)

        Returns:
            Dictionary with PII masked in specified keys

        Raises:
            RuntimeError: If masking any value fails
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str) and value and (keys_to_mask is None or key in keys_to_mask):
                result[key] = self.mask_pii(value)
            else:
                result[key] = value
        return result

    async def async_detect_pii(self, text: str) -> list[PII]:
        """Async wrapper for PII detection.

        Args:
            text: Text to scan for PII

        Returns:
            List of detected PII entities
        """
        return self.detect_pii(text)

    async def async_mask_pii(self, text: str) -> str:
        """Async wrapper for PII masking.

        Args:
            text: Text to mask

        Returns:
            Text with PII masked
        """
        return self.mask_pii(text)


# Singleton instance for use throughout the application
_pii_detector: PIIDetector | None = None


def get_pii_detector(config: PIIConfig | None = None) -> PIIDetector:
    """Get or create singleton PII detector instance.

    Args:
        config: PIIConfig object (only used on first call)

    Returns:
        PIIDetector instance
    """
    global _pii_detector
    if _pii_detector is None:
        _pii_detector = PIIDetector(config)
    return _pii_detector
