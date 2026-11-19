"""Pytest configuration for integration tests.

Presidio and the cross-encoder run for real here; only the LLM call is faked so
the suite needs no API key. Tests that need a real library are skipped (with the
reason shown) when it cannot be loaded, e.g. the reranker model download fails.
"""

from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import pytest

from aegis_platform.rag.access_control import (
    AccessLevel,
    DocumentAccessControl,
    DocumentMetadata,
    Role,
)
from aegis_platform.rag.cross_encoder_reranker import CrossEncoderReranker
from aegis_platform.rag.llm_query_enhancer import (
    LLMQueryEnhancer,
    reset_llm_query_enhancer_instance,
)
from aegis_platform.rag.pii_detector import PIIDetector

FAKE_EXPANSIONS = "1. data protection\n2. information security\n3. security controls"

CORPUS = [
    DocumentMetadata(
        doc_id="public_doc",
        title="Cafeteria",
        access_level=AccessLevel.PUBLIC,
        organization_id="org_123",
    ),
    DocumentMetadata(
        doc_id="internal_doc",
        title="Security Handbook",
        access_level=AccessLevel.INTERNAL,
        organization_id="org_123",
    ),
    DocumentMetadata(
        doc_id="analyst_doc",
        title="Data Classification",
        access_level=AccessLevel.RESTRICTED,
        allowed_roles=[Role.ANALYST],
        organization_id="org_123",
    ),
    DocumentMetadata(
        doc_id="restricted_doc",
        title="Board Minutes",
        access_level=AccessLevel.RESTRICTED,
        allowed_roles=[Role.ADMIN],
        organization_id="org_123",
    ),
    DocumentMetadata(
        doc_id="confidential_doc",
        title="Incident Postmortem",
        access_level=AccessLevel.CONFIDENTIAL,
        organization_id="org_123",
    ),
    DocumentMetadata(
        doc_id="org_b_doc",
        title="Org B Security Plan",
        access_level=AccessLevel.INTERNAL,
        organization_id="org_b",
    ),
]


@pytest.fixture(scope="session")
def pii_detector() -> PIIDetector:
    """Real Presidio-backed detector (not the shared singleton)."""
    try:
        return PIIDetector()
    except (ImportError, OSError) as e:
        pytest.skip(f"Presidio or spaCy model unavailable: {e}")


@pytest.fixture(scope="session")
def reranker() -> CrossEncoderReranker:
    """Real cross-encoder reranker with the model loaded once per session."""
    instance = CrossEncoderReranker()
    try:
        instance._ensure_model_loaded()
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"Cross-encoder model unavailable: {e}")
    return instance


@pytest.fixture
def fake_llm() -> Iterator[AsyncMock]:
    """Replace the enhancer's LLM call with a deterministic response."""
    reset_llm_query_enhancer_instance()
    with patch.object(
        LLMQueryEnhancer, "_call_llm", new=AsyncMock(return_value=FAKE_EXPANSIONS)
    ) as mock_call:
        yield mock_call
    reset_llm_query_enhancer_instance()


@pytest.fixture
def acl() -> Iterator[DocumentAccessControl]:
    """Fresh access-control singleton with the test corpus registered."""
    DocumentAccessControl.reset_instance()
    instance = DocumentAccessControl.get_instance()
    for doc in CORPUS:
        instance.register_document(doc)
    yield instance
    DocumentAccessControl.reset_instance()
