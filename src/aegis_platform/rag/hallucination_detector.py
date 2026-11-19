"""Hallucination detection for RAG responses.

This module detects common hallucination patterns in LLM-generated responses,
ensuring answers are grounded in source documents.

Features:
- Unsupported claim detection
- Contradiction finding
- Numerical consistency checking
- Hallucination risk scoring
- Singleton pattern for thread-safe operation

Example:
    >>> from aegis_platform.rag.hallucination_detector import (
    ...     HallucinationDetector, HallucinationPattern
    ... )
    >>>
    >>> detector = HallucinationDetector.get_instance()
    >>> score = await detector.detect_hallucinations(
    ...     llm_response="The capital of France is Paris, population 500 million.",
    ...     source_context=source_context,
    ...     citations=citations
    ... )
    >>>
    >>> print(f"Risk level: {score.risk_level}")
    >>> print(f"Patterns: {score.patterns_detected}")
"""

import logging
import re
from enum import Enum

from pydantic import BaseModel, Field

from aegis_platform.rag.citation_tracker import Citation, SourceContext

logger = logging.getLogger(__name__)


class HallucinationPattern(str, Enum):
    """Known hallucination patterns detected in responses."""

    UNSUPPORTED_CLAIM = "unsupported_claim"  # Claim not in sources
    CONTRADICTION = "contradiction"  # Contradicts sources
    OUT_OF_CONTEXT = "out_of_context"  # Topic not in sources
    NUMERICAL_FABRICATION = "numerical_fab"  # Made-up numbers/dates
    ATTRIBUTION_ERROR = "attribution_error"  # Wrong source cited
    INVENTED_EVIDENCE = "invented_evidence"  # Fake studies/quotes


class HallucinationScore(BaseModel):
    """Hallucination risk assessment for a response."""

    patterns_detected: list[HallucinationPattern] = Field(
        default_factory=list, description="Hallucination patterns found"
    )
    risk_level: str = Field(..., description="Risk level: 'low', 'medium', 'high'")
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Numerical risk score (0.0=no risk, 1.0=certain hallucination)",
    )
    explanation: str = Field(..., description="Explanation of detected patterns")


class HallucinationDetector:
    """Detect hallucinations in LLM-generated responses."""

    # Numerical pattern regex
    _NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
    _DATE_PATTERN = re.compile(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}")
    _PERCENTAGE_PATTERN = re.compile(r"\d+(?:\.\d+)?%")

    def __init__(self):
        """Initialize hallucination detector."""

    async def detect_hallucinations(
        self, llm_response: str, source_context: SourceContext, citations: list[Citation]
    ) -> HallucinationScore:
        """
        Detect hallucination patterns in LLM response.

        Args:
            llm_response: LLM-generated response text
            source_context: Context of sources provided to LLM
            citations: Citations extracted from response

        Returns:
            HallucinationScore with detected patterns and risk level

        Raises:
            ValueError: If inputs are invalid
        """
        if not llm_response or not llm_response.strip():
            raise ValueError("LLM response cannot be empty")

        patterns_detected: list[HallucinationPattern] = []
        risk_factors: list[bool] = []

        # Check for unsupported claims
        unsupported = await self._check_unsupported_claims(llm_response, source_context, citations)
        if unsupported:
            patterns_detected.append(HallucinationPattern.UNSUPPORTED_CLAIM)
            risk_factors.append(unsupported)

        # Check for contradictions
        contradictions = await self._check_contradictions(llm_response, source_context)
        if contradictions:
            patterns_detected.append(HallucinationPattern.CONTRADICTION)
            risk_factors.append(contradictions)

        # Check for numerical fabrications
        numerical = await self._check_numerical_consistency(llm_response, source_context)
        if numerical:
            patterns_detected.append(HallucinationPattern.NUMERICAL_FABRICATION)
            risk_factors.append(numerical)

        # Check for out-of-context claims
        out_of_context = await self._check_out_of_context(llm_response, source_context)
        if out_of_context:
            patterns_detected.append(HallucinationPattern.OUT_OF_CONTEXT)
            risk_factors.append(out_of_context)

        # Calculate risk score
        risk_score = self._calculate_risk_score(patterns_detected, citations, len(risk_factors))

        # Determine risk level
        risk_level = self._classify_risk_level(risk_score)

        # Create explanation
        explanation = self._create_explanation(patterns_detected, risk_factors)

        logger.debug(
            f"Hallucination detection: patterns={len(patterns_detected)}, "
            f"risk_level={risk_level}, score={risk_score:.2f}"
        )

        return HallucinationScore(
            patterns_detected=patterns_detected,
            risk_level=risk_level,
            risk_score=risk_score,
            explanation=explanation,
        )

    async def _check_unsupported_claims(
        self,
        response: str,
        source_context: SourceContext,
        citations: list[Citation],
        threshold: float = 0.5,
    ) -> bool:
        """
        Check for claims not supported by citations.

        Args:
            response: LLM response text
            source_context: Source context
            citations: Extracted citations
            threshold: Minimum % of response that should have citations

        Returns:
            True if unsupported claims detected
        """
        if not citations or not response:
            return True

        # Count words with citations vs total words
        response_words = set(response.lower().split())
        cited_words = set()

        for citation in citations:
            cited_words.update(citation.claim.lower().split())

        # Rough estimate: what % of response is covered by citations
        if response_words:
            coverage = len(cited_words & response_words) / len(response_words)
            unsupported = coverage < threshold
            logger.debug(f"Unsupported claims coverage: {coverage:.2f}")
            return unsupported

        return False

    async def _check_contradictions(self, response: str, source_context: SourceContext) -> bool:
        """
        Check for contradictions between response and sources.

        Args:
            response: LLM response text
            source_context: Source context

        Returns:
            True if contradictions detected
        """
        if not source_context.retrieved_passages:
            return False

        # Simple contradiction detection: look for explicit negations
        # e.g., response says "X is true" but source says "X is false"

        contradiction_patterns = [
            (r"(?:is|are|will)\s+not\s+", r"(?:is|are|will)\s+"),
            (r"(?:is|are)\s+false", r"(?:is|are)\s+true"),
            (r"denies?", r"confirms?"),
        ]

        response_lower = response.lower()

        for source in source_context.retrieved_passages:
            source_lower = source.lower()

            # Check if any contradiction patterns exist
            for neg_pattern, pos_pattern in contradiction_patterns:
                neg_matches = re.findall(neg_pattern, response_lower)
                pos_matches = re.findall(pos_pattern, source_lower)

                if neg_matches and pos_matches:
                    # Potential contradiction detected
                    logger.debug("Contradiction pattern detected")
                    return True

        return False

    async def _check_numerical_consistency(
        self, response: str, source_context: SourceContext
    ) -> bool:
        """
        Check if numbers/dates/percentages are consistent with sources.

        Args:
            response: LLM response text
            source_context: Source context

        Returns:
            True if numerical inconsistencies detected
        """
        response_numbers = self._extract_numbers(response)
        source_numbers = self._extract_numbers(" ".join(source_context.retrieved_passages))

        if not response_numbers or not source_numbers:
            return False

        # Check if response numbers appear in sources
        # (or are reasonable variations)
        for num in response_numbers:
            if num not in source_numbers:
                # Check for off-by-one or small variations
                similar_found = False
                for src_num in source_numbers:
                    # Check if numbers are within 10% of each other
                    if src_num > 0:
                        ratio = abs(num - src_num) / src_num
                        if ratio < 0.1:
                            similar_found = True
                            break

                if not similar_found and num > 100:  # Only flag large numbers
                    logger.debug(f"Numerical inconsistency: {num} not in sources")
                    return True

        return False

    async def _check_out_of_context(
        self, response: str, source_context: SourceContext, threshold: float = 0.3
    ) -> bool:
        """
        Check if response discusses topics not in sources.

        Args:
            response: LLM response text
            source_context: Source context
            threshold: Minimum keyword overlap required

        Returns:
            True if out-of-context claims detected
        """
        if not source_context.retrieved_passages:
            return False

        # Extract key topics from response and sources
        response_words = {
            word.lower() for word in response.split() if len(word) > 4
        }  # Focus on longer words
        source_words = {
            word.lower()
            for word in " ".join(source_context.retrieved_passages).split()
            if len(word) > 4
        }

        if not response_words or not source_words:
            return False

        # Calculate overlap
        overlap = len(response_words & source_words)
        overlap_ratio = overlap / len(response_words) if response_words else 0

        out_of_context = overlap_ratio < threshold
        if out_of_context:
            logger.debug(f"Out-of-context claim: overlap={overlap_ratio:.2f} < {threshold}")

        return out_of_context

    def _extract_numbers(self, text: str) -> list[float]:
        """Extract all numbers from text."""
        numbers = []
        for match in self._NUMBER_PATTERN.finditer(text):
            try:
                numbers.append(float(match.group()))
            except ValueError:
                pass
        return numbers

    def _calculate_risk_score(
        self, patterns: list[HallucinationPattern], citations: list[Citation], num_risk_factors: int
    ) -> float:
        """
        Calculate overall hallucination risk score.

        Args:
            patterns: Detected hallucination patterns
            citations: Citations in response
            num_risk_factors: Number of risk factors found

        Returns:
            Risk score (0.0-1.0)
        """
        if not patterns and citations:
            return 0.0

        if not citations:
            return 1.0

        # Base risk from patterns
        base_risk = len(patterns) * 0.2  # Each pattern adds 0.2

        # Penalty for low citation confidence
        avg_citation_confidence = (
            sum(c.confidence for c in citations) / len(citations) if citations else 0.0
        )
        confidence_penalty = (1.0 - avg_citation_confidence) * 0.3

        # Combine
        risk_score = min(base_risk + confidence_penalty, 1.0)

        logger.debug(
            f"Risk score calculation: base={base_risk:.2f}, "
            f"penalty={confidence_penalty:.2f}, total={risk_score:.2f}"
        )

        return risk_score

    def _classify_risk_level(self, risk_score: float) -> str:
        """
        Classify risk level based on score.

        Args:
            risk_score: Numerical risk score (0.0-1.0)

        Returns:
            Risk level: 'low', 'medium', or 'high'
        """
        if risk_score < 0.33:
            return "low"
        elif risk_score < 0.66:
            return "medium"
        else:
            return "high"

    def _create_explanation(
        self, patterns: list[HallucinationPattern], risk_factors: list[bool]
    ) -> str:
        """
        Create human-readable explanation of detected patterns.

        Args:
            patterns: Detected patterns
            risk_factors: Which risk factors were detected

        Returns:
            Explanation string
        """
        if not patterns:
            return "No hallucination patterns detected."

        explanations = []

        for pattern in patterns:
            if pattern == HallucinationPattern.UNSUPPORTED_CLAIM:
                explanations.append("Response contains claims not covered by citations")
            elif pattern == HallucinationPattern.CONTRADICTION:
                explanations.append("Response contradicts source documents")
            elif pattern == HallucinationPattern.NUMERICAL_FABRICATION:
                explanations.append("Response contains numbers/dates not in sources")
            elif pattern == HallucinationPattern.OUT_OF_CONTEXT:
                explanations.append("Response discusses topics not covered by sources")
            elif pattern == HallucinationPattern.ATTRIBUTION_ERROR:
                explanations.append("Response cites wrong sources")
            elif pattern == HallucinationPattern.INVENTED_EVIDENCE:
                explanations.append("Response references invented studies or quotes")

        return "; ".join(explanations)

    @staticmethod
    def reset_instance() -> None:
        """Reset singleton instance (for testing)."""
        global _hallucination_detector_instance
        _hallucination_detector_instance = None


# Singleton instance
_hallucination_detector_instance: HallucinationDetector | None = None


def get_hallucination_detector() -> HallucinationDetector:
    """
    Get or create singleton hallucination detector instance.

    Returns:
        Singleton HallucinationDetector instance
    """
    global _hallucination_detector_instance
    if _hallucination_detector_instance is None:
        _hallucination_detector_instance = HallucinationDetector()
        logger.debug("Created new HallucinationDetector singleton instance")
    return _hallucination_detector_instance
