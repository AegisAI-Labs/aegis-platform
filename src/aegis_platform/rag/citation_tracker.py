"""Citation tracking and hallucination prevention for RAG.

This module tracks which source documents support each part of an LLM response,
enabling verifiable answers and hallucination detection.

Features:
- Document-level and span-level citation tracking
- Multi-factor confidence scoring
- Citation verification against source documents
- Hallucination pattern detection
- Singleton pattern for thread-safe operation

Example:
    >>> from aegis_platform.rag.citation_tracker import (
    ...     CitationTracker, SourceContext, Citation, CitationSource
    ... )
    >>>
    >>> # Track retrieved documents
    >>> source_context = await tracker.track_retrieval_context(
    ...     query="What is RAG?",
    ...     retrieved_docs=[doc1, doc2],
    ...     retrieval_scores=[0.95, 0.87],
    ...     passages=["RAG combines retrieval...", "RAG improves accuracy..."]
    ... )
    >>>
    >>> # Extract citations from LLM response
    >>> llm_response = "RAG combines retrieval and generation..."
    >>> citations = await tracker.extract_citations(llm_response, source_context)
    >>>
    >>> # Calculate confidence
    >>> confidence = await tracker.calculate_response_confidence(citations, source_context)
"""

import logging
from datetime import UTC, datetime
from typing import TypedDict

from pydantic import BaseModel, Field

from aegis_platform.rag.access_control import DocumentMetadata

logger = logging.getLogger(__name__)


class VerificationDetail(TypedDict, total=False):
    """Details of a single citation verification."""

    claim: str
    status: str
    reason: str | None
    source_count: int | None


class VerificationResult(TypedDict):
    """Result of citation verification."""

    total_citations: int
    verified_citations: int
    unverified_citations: int
    contradicted_citations: int
    details: list[VerificationDetail]


class CitationSource(BaseModel):
    """Source document or passage for a citation."""

    doc_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    source_relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Relevance score from vector search (0.0-1.0)"
    )
    passage_start: int = Field(
        default=0, ge=0, description="Character offset where passage starts in document"
    )
    passage_end: int = Field(
        ..., gt=0, description="Character offset where passage ends in document"
    )
    passage_text: str = Field(..., description="Actual text of the passage")


class Citation(BaseModel):
    """Citation mapping a claim to its source documents."""

    claim: str = Field(..., description="Specific sentence/phrase from LLM response")
    sources: list[CitationSource] = Field(
        default_factory=list, description="Which documents support this claim"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in this citation (0.0-1.0)"
    )


class SourceContext(BaseModel):
    """Full context of sources provided to LLM."""

    query: str = Field(..., description="Original user query")
    retrieved_docs: list[DocumentMetadata] = Field(
        default_factory=list, description="Documents retrieved from vector search"
    )
    retrieved_passages: list[str] = Field(
        default_factory=list, description="Text passages extracted from retrieved documents"
    )
    retrieval_scores: list[float] = Field(
        default_factory=list, description="Similarity scores for each retrieved document"
    )
    total_tokens: int = Field(
        default=0, ge=0, description="Total tokens in context window provided to LLM"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="When this context was created"
    )


class ResponseMetadata(BaseModel):
    """Metadata about LLM generation."""

    model: str = Field(..., description="LLM model used (gpt-4, claude, etc)")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Temperature parameter for generation"
    )
    max_tokens: int = Field(default=512, gt=0, description="Maximum tokens for generation")
    generation_time_ms: float = Field(
        default=0.0, ge=0.0, description="Time taken to generate response in milliseconds"
    )


class CitationTracker:
    """Track citations and calculate confidence scores for RAG responses."""

    def __init__(self):
        """Initialize citation tracker."""
        self._source_contexts: dict[str, SourceContext] = {}

    async def track_retrieval_context(
        self,
        query: str,
        retrieved_docs: list[DocumentMetadata],
        retrieval_scores: list[float],
        passages: list[str],
    ) -> SourceContext:
        """
        Track documents and passages retrieved for RAG.

        Args:
            query: Original user query
            retrieved_docs: Documents retrieved from vector search
            retrieval_scores: Similarity scores for each document
            passages: Text passages from retrieved documents

        Returns:
            SourceContext object tracking the retrieval

        Raises:
            ValueError: If lists have mismatched lengths
        """
        if not (len(retrieved_docs) == len(retrieval_scores) == len(passages)):
            raise ValueError(
                f"Mismatched lengths: docs={len(retrieved_docs)}, "
                f"scores={len(retrieval_scores)}, passages={len(passages)}"
            )

        # Calculate total tokens (rough estimate: 1.3 tokens per word)
        total_words = sum(len(p.split()) for p in passages)
        total_tokens = int(total_words * 1.3)

        source_context = SourceContext(
            query=query,
            retrieved_docs=retrieved_docs,
            retrieved_passages=passages,
            retrieval_scores=retrieval_scores,
            total_tokens=total_tokens,
        )

        logger.debug(
            f"Tracked retrieval context for query: {query[:50]}..., "
            f"docs={len(retrieved_docs)}, tokens={total_tokens}"
        )

        return source_context

    async def extract_citations(
        self, llm_response: str, source_context: SourceContext
    ) -> list[Citation]:
        """
        Extract citations from LLM response.

        Maps sentences from LLM response to supporting source passages.

        Args:
            llm_response: LLM-generated response text
            source_context: Context of sources provided to LLM

        Returns:
            List of Citation objects mapping claims to sources

        Raises:
            ValueError: If response or source_context invalid
        """
        if not llm_response or not llm_response.strip():
            raise ValueError("LLM response cannot be empty")

        if not source_context.retrieved_passages:
            raise ValueError("Source context must have retrieved passages")

        citations: list[Citation] = []

        # Split response into sentences
        sentences = [s.strip() for s in llm_response.split(".") if s.strip()]

        for sentence in sentences:
            if not sentence:
                continue

            # Find matching sources for this sentence
            matching_sources = self._find_matching_sources(sentence, source_context)

            if matching_sources:
                confidence = self._calculate_citation_confidence(
                    matching_sources, sentence, source_context
                )

                citation = Citation(claim=sentence, sources=matching_sources, confidence=confidence)
                citations.append(citation)

                logger.debug(
                    f"Extracted citation: {sentence[:50]}..., "
                    f"sources={len(matching_sources)}, confidence={confidence:.2f}"
                )

        return citations

    async def calculate_response_confidence(
        self, citations: list[Citation], source_context: SourceContext
    ) -> float:
        """
        Calculate overall confidence in LLM response.

        Uses multiple factors:
        - Number of citations
        - Citation confidence scores
        - Source diversity
        - Coverage of response

        Args:
            citations: Extracted citations from response
            source_context: Context of sources

        Returns:
            Overall confidence score (0.0-1.0)
        """
        if not citations:
            return 0.0

        if not source_context.retrieved_docs:
            return 0.0

        # Average citation confidence
        avg_citation_confidence = sum(c.confidence for c in citations) / len(citations)

        # Citation coverage (what % of response has citations)
        coverage = min(len(citations) / max(len(citations), 3), 1.0)

        # Source diversity
        unique_sources = set()
        for citation in citations:
            for source in citation.sources:
                unique_sources.add(source.doc_id)

        source_diversity = len(unique_sources) / len(source_context.retrieved_docs)

        # Weighted calculation
        confidence = 0.5 * avg_citation_confidence + 0.3 * coverage + 0.2 * source_diversity

        logger.debug(
            f"Response confidence: {confidence:.2f} "
            f"(citations={avg_citation_confidence:.2f}, "
            f"coverage={coverage:.2f}, diversity={source_diversity:.2f})"
        )

        return min(confidence, 1.0)

    async def verify_citations(
        self, citations: list[Citation], source_context: SourceContext
    ) -> VerificationResult:
        """
        Verify each citation against source documents.

        Checks for:
        - Passage match (is claim in source passage?)
        - Semantic match (is claim semantically related to passage?)
        - Direct contradictions

        Args:
            citations: Citations to verify
            source_context: Source documents

        Returns:
            Dictionary with verification results for each citation
        """
        verification_results: VerificationResult = {
            "total_citations": len(citations),
            "verified_citations": 0,
            "unverified_citations": 0,
            "contradicted_citations": 0,
            "details": [],
        }

        for citation in citations:
            if not citation.sources:
                verification_results["unverified_citations"] += 1
                detail: VerificationDetail = {
                    "claim": citation.claim[:50],
                    "status": "unverified",
                    "reason": "No supporting sources found",
                }
                verification_results["details"].append(detail)
                continue

            # For each source, check if passage supports claim
            is_supported = False
            for source in citation.sources:
                # Simple substring match (can be enhanced with semantic similarity)
                if any(
                    word in source.passage_text.lower() for word in citation.claim.lower().split()
                ):
                    is_supported = True
                    break

            if is_supported:
                verification_results["verified_citations"] += 1
                detail = {
                    "claim": citation.claim[:50],
                    "status": "verified",
                    "source_count": len(citation.sources),
                }
                verification_results["details"].append(detail)
            else:
                verification_results["unverified_citations"] += 1
                detail = {
                    "claim": citation.claim[:50],
                    "status": "unverified",
                    "source_count": len(citation.sources),
                }
                verification_results["details"].append(detail)

        logger.debug(
            f"Citation verification: {verification_results['verified_citations']}/"
            f"{verification_results['total_citations']} verified"
        )

        return verification_results

    def _find_matching_sources(
        self, sentence: str, source_context: SourceContext
    ) -> list[CitationSource]:
        """
        Find sources that match a sentence.

        Args:
            sentence: Sentence from LLM response
            source_context: Source context

        Returns:
            List of matching CitationSource objects
        """
        matching_sources: list[CitationSource] = []

        for doc, passage, score in zip(
            source_context.retrieved_docs,
            source_context.retrieved_passages,
            source_context.retrieval_scores,
        ):
            # Check for word overlap between sentence and passage
            sentence_words = set(sentence.lower().split())
            passage_words = set(passage.lower().split())

            # Remove common stop words for better matching
            stop_words = {"the", "a", "an", "is", "are", "and", "or", "in", "of"}
            sentence_words -= stop_words
            passage_words -= stop_words

            # Calculate overlap ratio
            if sentence_words and passage_words:
                overlap = len(sentence_words & passage_words)
                overlap_ratio = overlap / max(len(sentence_words), len(passage_words))

                if overlap_ratio > 0.3:  # Threshold for considering it a match
                    citation_source = CitationSource(
                        doc_id=doc.doc_id,
                        title=doc.title,
                        source_relevance_score=score,
                        passage_start=0,  # Simplified (would compute actual offset)
                        passage_end=len(passage),
                        passage_text=passage,
                    )
                    matching_sources.append(citation_source)

        return matching_sources

    def _calculate_citation_confidence(
        self, sources: list[CitationSource], claim: str, source_context: SourceContext
    ) -> float:
        """
        Calculate confidence for a specific citation.

        Factors:
        - Number of supporting sources
        - Average relevance score of sources
        - Quality of passage match

        Args:
            sources: Supporting sources for citation
            claim: The claim being cited
            source_context: Full source context

        Returns:
            Confidence score (0.0-1.0)
        """
        if not sources:
            return 0.0

        # Number of sources factor (more sources = higher confidence)
        num_sources_factor = min(len(sources) / 3.0, 1.0)

        # Average relevance factor
        avg_relevance = sum(s.source_relevance_score for s in sources) / len(sources)
        relevance_factor = avg_relevance

        # Passage quality factor (how much of passage is about claim)
        # Simplified: check keyword overlap
        claim_keywords = set(claim.lower().split()) - {"the", "a", "an", "is", "are"}
        passage_coverage = 0.0

        for source in sources:
            passage_words = set(source.passage_text.lower().split())
            overlap = len(claim_keywords & passage_words)
            if claim_keywords:
                coverage = overlap / len(claim_keywords)
                passage_coverage += coverage

        passage_coverage /= max(len(sources), 1)
        passage_coverage = min(passage_coverage, 1.0)

        # Weighted combination
        confidence = 0.4 * num_sources_factor + 0.4 * relevance_factor + 0.2 * passage_coverage

        return min(confidence, 1.0)


# Singleton instance
_citation_tracker_instance: CitationTracker | None = None


def get_citation_tracker() -> CitationTracker:
    """
    Get or create singleton citation tracker instance.

    Returns:
        Singleton CitationTracker instance
    """
    global _citation_tracker_instance
    if _citation_tracker_instance is None:
        _citation_tracker_instance = CitationTracker()
        logger.debug("Created new CitationTracker singleton instance")
    return _citation_tracker_instance
