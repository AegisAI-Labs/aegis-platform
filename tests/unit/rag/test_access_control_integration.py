"""Integration tests for Document Access Control within RAG pipeline.

Tests cover:
- Access control before RAG query
- Access control after RAG retrieval
- Multi-tenant isolation
- Audit logging in RAG context
- Admin override
"""

import pytest

from aegis_platform.rag.access_control import (
    AccessControlConfig,
    AccessLevel,
    DocumentAccessControl,
    DocumentMetadata,
    Role,
    UserContext,
)


class TestAccessControlBeforeQuery:
    """Test access control filtering before RAG query."""

    @pytest.fixture
    def setup_rag_scenario(self):
        """Set up a RAG scenario with multiple documents."""
        DocumentAccessControl.reset_instance()
        ac = DocumentAccessControl()

        # Create test documents with different access levels
        docs = [
            DocumentMetadata(
                doc_id="financial_report",
                title="Q4 Financial Report",
                access_level=AccessLevel.RESTRICTED,
                allowed_roles=[Role.ANALYST, Role.ADMIN],
            ),
            DocumentMetadata(
                doc_id="public_news", title="Company News", access_level=AccessLevel.PUBLIC
            ),
            DocumentMetadata(
                doc_id="research_paper",
                title="Market Research",
                access_level=AccessLevel.RESTRICTED,
                allowed_roles=[Role.RESEARCHER, Role.ADMIN],
            ),
            DocumentMetadata(
                doc_id="internal_memo", title="Executive Memo", access_level=AccessLevel.INTERNAL
            ),
            DocumentMetadata(
                doc_id="board_minutes",
                title="Board Meeting Minutes",
                access_level=AccessLevel.CONFIDENTIAL,
            ),
        ]

        for doc in docs:
            ac.register_document(doc)

        return ac

    @pytest.mark.asyncio
    async def test_analyst_query_filtering(self, setup_rag_scenario):
        """Test analyst user gets correctly filtered results."""
        ac = setup_rag_scenario
        analyst = UserContext(user_id="analyst1", role=Role.ANALYST)

        # Simulate vector search returning all documents
        raw_results = [
            {"doc_id": "financial_report", "score": 0.95},
            {"doc_id": "public_news", "score": 0.85},
            {"doc_id": "research_paper", "score": 0.75},
            {"doc_id": "internal_memo", "score": 0.65},
            {"doc_id": "board_minutes", "score": 0.55},
        ]

        # Filter results by access control
        filtered = ac.filter_search_results(analyst, raw_results)

        # Analyst should see: financial_report (allowed), public_news (public),
        # internal_memo (authenticated), but not research_paper or board_minutes
        accessible_docs = {r["doc_id"] for r in filtered}

        assert "financial_report" in accessible_docs  # Analyst role
        assert "public_news" in accessible_docs  # Public
        assert "internal_memo" in accessible_docs  # Internal
        assert "research_paper" not in accessible_docs  # Researcher only
        assert "board_minutes" not in accessible_docs  # Admin only

    @pytest.mark.asyncio
    async def test_guest_query_filtering(self, setup_rag_scenario):
        """Test guest user has minimal access."""
        ac = setup_rag_scenario
        guest = UserContext(user_id="guest1", role=Role.GUEST)

        raw_results = [
            {"doc_id": "financial_report", "score": 0.95},
            {"doc_id": "public_news", "score": 0.85},
            {"doc_id": "research_paper", "score": 0.75},
            {"doc_id": "internal_memo", "score": 0.65},
            {"doc_id": "board_minutes", "score": 0.55},
        ]

        filtered = ac.filter_search_results(guest, raw_results)

        # Guest should only see public documents
        accessible_docs = {r["doc_id"] for r in filtered}

        assert "public_news" in accessible_docs  # Public
        assert "financial_report" not in accessible_docs
        assert "research_paper" not in accessible_docs
        assert "internal_memo" not in accessible_docs
        assert "board_minutes" not in accessible_docs

    @pytest.mark.asyncio
    async def test_admin_full_query_access(self, setup_rag_scenario):
        """Test admin user gets all results."""
        ac = setup_rag_scenario
        admin = UserContext(user_id="admin1", role=Role.ADMIN)

        raw_results = [
            {"doc_id": "financial_report", "score": 0.95},
            {"doc_id": "public_news", "score": 0.85},
            {"doc_id": "research_paper", "score": 0.75},
            {"doc_id": "internal_memo", "score": 0.65},
            {"doc_id": "board_minutes", "score": 0.55},
        ]

        filtered = ac.filter_search_results(admin, raw_results)

        # Admin should see all documents
        assert len(filtered) == 5


class TestAccessControlAfterRetrieval:
    """Test access control as post-processing filter."""

    @pytest.fixture
    def setup_retrieval_scenario(self):
        """Set up scenario with retrieved documents."""
        DocumentAccessControl.reset_instance()
        ac = DocumentAccessControl()

        docs = [
            DocumentMetadata(
                doc_id="doc1", title="Public Article", access_level=AccessLevel.PUBLIC
            ),
            DocumentMetadata(
                doc_id="doc2", title="Internal Report", access_level=AccessLevel.INTERNAL
            ),
            DocumentMetadata(
                doc_id="doc3",
                title="Restricted Data",
                access_level=AccessLevel.RESTRICTED,
                allowed_roles=[Role.ANALYST],
            ),
        ]

        for doc in docs:
            ac.register_document(doc)

        return ac

    @pytest.mark.asyncio
    async def test_post_retrieval_filtering(self, setup_retrieval_scenario):
        """Test filtering documents after retrieval."""
        ac = setup_retrieval_scenario
        user = UserContext(user_id="user1", role=Role.USER)

        # Retrieved documents with context excerpts
        retrieved = [
            {"doc_id": "doc1", "content": "This is public content", "score": 0.9},
            {"doc_id": "doc2", "content": "This is internal content", "score": 0.8},
            {"doc_id": "doc3", "content": "This is restricted content", "score": 0.7},
        ]

        # Filter before returning to user
        accessible = ac.filter_search_results(user, retrieved)

        # User should get public and internal only
        assert len(accessible) == 2
        doc_ids = {doc["doc_id"] for doc in accessible}
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
        assert "doc3" not in doc_ids


class TestMultiTenantIsolation:
    """Test multi-tenant document isolation."""

    @pytest.fixture
    def setup_multitenant_scenario(self):
        """Set up multi-tenant environment."""
        DocumentAccessControl.reset_instance()
        ac = DocumentAccessControl()

        # Org A documents
        docs_a = [
            DocumentMetadata(
                doc_id="org_a_doc1",
                title="Org A Document 1",
                access_level=AccessLevel.INTERNAL,
                organization_id="org_a",
            ),
            DocumentMetadata(
                doc_id="org_a_doc2",
                title="Org A Document 2",
                access_level=AccessLevel.RESTRICTED,
                allowed_roles=[Role.ANALYST],
                organization_id="org_a",
            ),
        ]

        # Org B documents
        docs_b = [
            DocumentMetadata(
                doc_id="org_b_doc1",
                title="Org B Document 1",
                access_level=AccessLevel.INTERNAL,
                organization_id="org_b",
            ),
        ]

        for doc in docs_a + docs_b:
            ac.register_document(doc)

        return ac

    @pytest.mark.asyncio
    async def test_org_isolation_check_access(self, setup_multitenant_scenario):
        """Test users can't access documents from other organizations."""
        ac = setup_multitenant_scenario

        user_a = UserContext(user_id="user_a", role=Role.ANALYST, organization_id="org_a")

        # User A can access their org documents
        result = await ac.check_access(user_a, "org_a_doc1")
        assert result.is_allowed is True

        # User A cannot access other org documents
        result = await ac.check_access(user_a, "org_b_doc1")
        assert result.is_allowed is False

    def test_org_isolation_filter_results(self, setup_multitenant_scenario):
        """Test result filtering respects organization boundaries."""
        ac = setup_multitenant_scenario

        user_a = UserContext(user_id="user_a", role=Role.ANALYST, organization_id="org_a")

        # Mixed org search results
        raw_results = [
            {"doc_id": "org_a_doc1"},
            {"doc_id": "org_b_doc1"},
            {"doc_id": "org_a_doc2"},
        ]

        filtered = ac.filter_search_results(user_a, raw_results)

        # Only org_a documents should remain
        doc_ids = {r["doc_id"] for r in filtered}
        assert "org_a_doc1" in doc_ids
        assert "org_a_doc2" in doc_ids
        assert "org_b_doc1" not in doc_ids


class TestAuditLoggingIntegration:
    """Test audit logging in RAG context."""

    @pytest.fixture
    def setup_audit_scenario(self):
        """Set up scenario for audit testing."""
        DocumentAccessControl.reset_instance()
        config = AccessControlConfig(audit_enabled=True)
        ac = DocumentAccessControl(config)

        docs = [
            DocumentMetadata(doc_id="public_doc", title="Public", access_level=AccessLevel.PUBLIC),
            DocumentMetadata(
                doc_id="restricted_doc",
                title="Restricted",
                access_level=AccessLevel.RESTRICTED,
                allowed_roles=[Role.ANALYST],
            ),
        ]

        for doc in docs:
            ac.register_document(doc)

        return ac

    @pytest.mark.asyncio
    async def test_audit_logs_allowed_access(self, setup_audit_scenario):
        """Test audit log captures allowed access."""
        ac = setup_audit_scenario
        user = UserContext(user_id="user1", role=Role.USER)

        result = await ac.check_access(user, "public_doc", operation="read")

        assert result.audit_log_entry is not None
        assert result.audit_log_entry["user_id"] == "user1"
        assert result.audit_log_entry["resource_id"] == "public_doc"
        assert result.audit_log_entry["operation"] == "read"
        assert result.audit_log_entry["result"] == "allowed"

    @pytest.mark.asyncio
    async def test_audit_logs_denied_access(self, setup_audit_scenario):
        """Test audit log captures denied access."""
        ac = setup_audit_scenario
        user = UserContext(user_id="user1", role=Role.USER)

        result = await ac.check_access(user, "restricted_doc", operation="read")

        assert result.audit_log_entry is not None
        assert result.audit_log_entry["result"] == "denied"
        assert result.audit_log_entry["risk_level"] == "medium"


class TestAdminOverride:
    """Test admin override behavior."""

    @pytest.fixture
    def setup_admin_scenario(self):
        """Set up scenario for admin testing."""
        DocumentAccessControl.reset_instance()
        ac = DocumentAccessControl()

        docs = [
            DocumentMetadata(
                doc_id="confidential_doc",
                title="Confidential",
                access_level=AccessLevel.CONFIDENTIAL,
            ),
            DocumentMetadata(
                doc_id="restricted_doc",
                title="Restricted",
                access_level=AccessLevel.RESTRICTED,
                allowed_roles=[Role.ANALYST],
            ),
        ]

        for doc in docs:
            ac.register_document(doc)

        return ac

    @pytest.mark.asyncio
    async def test_admin_override_confidential(self, setup_admin_scenario):
        """Test admin can access confidential documents."""
        ac = setup_admin_scenario
        admin = UserContext(user_id="admin1", role=Role.ADMIN)

        result = await ac.check_access(admin, "confidential_doc")
        assert result.is_allowed is True

    @pytest.mark.asyncio
    async def test_admin_override_restricted(self, setup_admin_scenario):
        """Test admin can access any restricted document."""
        ac = setup_admin_scenario
        admin = UserContext(user_id="admin1", role=Role.ADMIN)
        researcher = UserContext(user_id="researcher1", role=Role.RESEARCHER)

        # Analyst doc restricted to analysts only
        result_admin = await ac.check_access(admin, "restricted_doc")
        result_researcher = await ac.check_access(researcher, "restricted_doc")

        assert result_admin.is_allowed is True
        assert result_researcher.is_allowed is False
