"""Unit tests for citation tracking module.

Tests cover:
- Citation source creation and validation
- Citation creation and scoring
- Source context tracking
- Citation extraction from LLM responses
- Confidence calculation
- Citation verification
- Singleton pattern
"""

import pytest

from aegis_platform.rag.access_control import (
    AccessLevel,
    DocumentMetadata,
)
from aegis_platform.rag.citation_tracker import (
    Citation,
    CitationSource,
    CitationTracker,
    ResponseMetadata,
    SourceContext,
    get_citation_tracker,
)


class TestCitationSource:
    """Test CitationSource model."""

    def test_citation_source_creation(self):
        """Test creating citation source."""
        source = CitationSource(
            doc_id="doc1",
            title="Test Document",
            source_relevance_score=0.95,
            passage_start=0,
            passage_end=100,
            passage_text="This is a test passage",
        )
        assert source.doc_id == "doc1"
        assert source.title == "Test Document"
        assert source.source_relevance_score == 0.95

    def test_citation_source_invalid_score(self):
        """Test that invalid scores are rejected."""
        with pytest.raises(ValueError):
            CitationSource(
                doc_id="doc1",
                title="Test",
                source_relevance_score=1.5,  # > 1.0
                passage_end=100,
                passage_text="Test",
            )

    def test_citation_source_negative_passage_end(self):
        """Test that negative passage end is rejected."""
        with pytest.raises(ValueError):
            CitationSource(
                doc_id="doc1",
                title="Test",
                source_relevance_score=0.9,
                passage_end=-1,  # Must be > 0
                passage_text="Test",
            )


class TestCitation:
    """Test Citation model."""

    def test_citation_creation(self):
        """Test creating citation with sources."""
        source = CitationSource(
            doc_id="doc1",
            title="Test",
            source_relevance_score=0.9,
            passage_end=100,
            passage_text="Test passage",
        )
        citation = Citation(claim="This is a test claim", sources=[source], confidence=0.85)
        assert citation.claim == "This is a test claim"
        assert len(citation.sources) == 1
        assert citation.confidence == 0.85

    def test_citation_default_confidence(self):
        """Test that confidence defaults to 0.0."""
        citation = Citation(claim="Test claim")
        assert citation.confidence == 0.0

    def test_citation_invalid_confidence(self):
        """Test that invalid confidence is rejected."""
        with pytest.raises(ValueError):
            Citation(claim="Test", confidence=1.5)  # > 1.0


class TestSourceContext:
    """Test SourceContext model."""

    def test_source_context_creation(self):
        """Test creating source context."""
        doc = DocumentMetadata(doc_id="doc1", title="Test", access_level=AccessLevel.PUBLIC)
        context = SourceContext(
            query="What is RAG?",
            retrieved_docs=[doc],
            retrieved_passages=["RAG combines retrieval and generation"],
            retrieval_scores=[0.95],
            total_tokens=50,
        )
        assert context.query == "What is RAG?"
        assert len(context.retrieved_docs) == 1
        assert context.total_tokens == 50

    def test_source_context_defaults(self):
        """Test that source context has proper defaults."""
        context = SourceContext(query="Test")
        assert context.retrieved_docs == []
        assert context.retrieved_passages == []
        assert context.retrieval_scores == []
        assert context.total_tokens == 0
        assert context.created_at is not None


class TestResponseMetadata:
    """Test ResponseMetadata model."""

    def test_response_metadata_creation(self):
        """Test creating response metadata."""
        metadata = ResponseMetadata(
            model="gpt-4", temperature=0.7, max_tokens=512, generation_time_ms=150.5
        )
        assert metadata.model == "gpt-4"
        assert metadata.generation_time_ms == 150.5

    def test_response_metadata_defaults(self):
        """Test response metadata defaults."""
        metadata = ResponseMetadata(model="claude")
        assert metadata.temperature == 0.7
        assert metadata.max_tokens == 512


class TestRetrievalTracking:
    """Test tracking retrieved documents."""

    @pytest.mark.asyncio
    async def test_track_retrieval_context(self):
        """Test tracking retrieval context."""
        tracker = CitationTracker()
        docs = [
            DocumentMetadata(doc_id="doc1", title="Doc 1"),
            DocumentMetadata(doc_id="doc2", title="Doc 2"),
        ]
        passages = ["Passage 1 content", "Passage 2 content"]
        scores = [0.95, 0.87]

        context = await tracker.track_retrieval_context(
            query="Test query", retrieved_docs=docs, retrieval_scores=scores, passages=passages
        )

        assert context.query == "Test query"
        assert len(context.retrieved_docs) == 2
        assert context.total_tokens > 0

    @pytest.mark.asyncio
    async def test_track_retrieval_mismatched_lengths(self):
        """Test that mismatched input lengths are rejected."""
        tracker = CitationTracker()
        docs = [DocumentMetadata(doc_id="doc1", title="Doc 1")]
        passages = ["Passage 1", "Passage 2"]  # Length mismatch
        scores = [0.95]

        with pytest.raises(ValueError, match="Mismatched lengths"):
            await tracker.track_retrieval_context(
                query="Test", retrieved_docs=docs, retrieval_scores=scores, passages=passages
            )


class TestCitationExtraction:
    """Test extracting citations from LLM responses."""

    @pytest.mark.asyncio
    async def test_extract_citations(self):
        """Test extracting citations from response."""
        tracker = CitationTracker()
        docs = [DocumentMetadata(doc_id="doc1", title="Test Doc")]
        passages = ["RAG combines retrieval and generation to improve accuracy"]
        scores = [0.95]

        context = await tracker.track_retrieval_context(
            query="What is RAG?", retrieved_docs=docs, retrieval_scores=scores, passages=passages
        )

        response = "RAG combines retrieval and generation. This improves accuracy."
        citations = await tracker.extract_citations(response, context)

        assert len(citations) > 0
        assert all(isinstance(c, Citation) for c in citations)

    @pytest.mark.asyncio
    async def test_extract_citations_empty_response(self):
        """Test that empty response raises error."""
        tracker = CitationTracker()
        context = SourceContext(query="Test")

        with pytest.raises(ValueError, match="cannot be empty"):
            await tracker.extract_citations("", context)

    @pytest.mark.asyncio
    async def test_extract_citations_no_passages(self):
        """Test that context without passages raises error."""
        tracker = CitationTracker()
        context = SourceContext(query="Test", retrieved_passages=[])

        with pytest.raises(ValueError, match="must have retrieved passages"):
            await tracker.extract_citations("Test response", context)


class TestConfidenceCalculation:
    """Test confidence calculation for citations."""

    @pytest.mark.asyncio
    async def test_calculate_response_confidence(self):
        """Test calculating response confidence."""
        tracker = CitationTracker()
        docs = [
            DocumentMetadata(doc_id="doc1", title="Doc 1"),
            DocumentMetadata(doc_id="doc2", title="Doc 2"),
        ]
        passages = ["Passage 1", "Passage 2"]
        scores = [0.95, 0.87]

        context = await tracker.track_retrieval_context(
            query="Test", retrieved_docs=docs, retrieval_scores=scores, passages=passages
        )

        # Create citations
        source = CitationSource(
            doc_id="doc1",
            title="Doc 1",
            source_relevance_score=0.95,
            passage_end=50,
            passage_text="Passage 1",
        )
        citation = Citation(claim="Test claim", sources=[source], confidence=0.85)

        confidence = await tracker.calculate_response_confidence([citation], context)

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.0

    @pytest.mark.asyncio
    async def test_calculate_confidence_no_citations(self):
        """Test confidence with no citations is 0.0."""
        tracker = CitationTracker()
        context = SourceContext(query="Test")

        confidence = await tracker.calculate_response_confidence([], context)
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_calculate_confidence_no_sources(self):
        """Test confidence with no source docs is 0.0."""
        tracker = CitationTracker()
        context = SourceContext(query="Test")

        citation = Citation(claim="Test")
        confidence = await tracker.calculate_response_confidence([citation], context)
        assert confidence == 0.0


class TestCitationVerification:
    """Test verifying citations against sources."""

    @pytest.mark.asyncio
    async def test_verify_citations(self):
        """Test verifying citations."""
        tracker = CitationTracker()
        docs = [DocumentMetadata(doc_id="doc1", title="Test")]
        passages = ["This is a test passage with specific words"]
        scores = [0.95]

        context = await tracker.track_retrieval_context(
            query="Test", retrieved_docs=docs, retrieval_scores=scores, passages=passages
        )

        source = CitationSource(
            doc_id="doc1",
            title="Test",
            source_relevance_score=0.95,
            passage_end=50,
            passage_text="This is a test passage",
        )
        citation = Citation(claim="This is a test passage", sources=[source], confidence=0.9)

        results = await tracker.verify_citations([citation], context)

        assert results["total_citations"] == 1
        assert results["verified_citations"] + results["unverified_citations"] == 1

    @pytest.mark.asyncio
    async def test_verify_citations_no_sources(self):
        """Test verifying citations with no sources."""
        tracker = CitationTracker()
        context = SourceContext(query="Test")

        citation = Citation(claim="Test claim", sources=[])
        results = await tracker.verify_citations([citation], context)

        assert results["unverified_citations"] == 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_track_empty_passages(self):
        """Test tracking with empty passages."""
        tracker = CitationTracker()
        docs = [DocumentMetadata(doc_id="doc1", title="Doc")]
        passages = [""]  # Empty passage
        scores = [0.95]

        context = await tracker.track_retrieval_context(
            query="Test", retrieved_docs=docs, retrieval_scores=scores, passages=passages
        )

        assert context.query == "Test"

    @pytest.mark.asyncio
    async def test_extract_citations_very_long_response(self):
        """Test extracting citations from long response."""
        tracker = CitationTracker()
        # Need matching lengths: 1 doc, 1 passage, 1 score
        docs = [DocumentMetadata(doc_id="doc1", title="Doc")]
        passages = ["Test passage content for all sentences"]
        scores = [0.9]

        context = await tracker.track_retrieval_context(
            query="Test", retrieved_docs=docs, retrieval_scores=scores, passages=passages
        )

        # Create a long response with many sentences
        response = ". ".join(["This is sentence number " + str(i) for i in range(50)])

        citations = await tracker.extract_citations(response, context)
        # Should handle long responses gracefully
        assert isinstance(citations, list)


class TestSingletonPattern:
    """Test singleton pattern for CitationTracker."""

    def test_singleton_creation(self):
        """Test that singleton is created once."""
        tracker1 = get_citation_tracker()
        tracker2 = get_citation_tracker()

        assert tracker1 is tracker2

    @pytest.mark.asyncio
    async def test_singleton_usage(self):
        """Test using singleton across multiple calls."""
        tracker1 = get_citation_tracker()

        docs = [DocumentMetadata(doc_id="doc1", title="Doc")]
        passages = ["Test passage"]
        scores = [0.95]

        await tracker1.track_retrieval_context(
            query="Query 1", retrieved_docs=docs, retrieval_scores=scores, passages=passages
        )

        # Get singleton again
        tracker2 = get_citation_tracker()
        assert tracker1 is tracker2
