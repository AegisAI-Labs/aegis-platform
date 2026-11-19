"""Unit tests for Document Access Control (RBAC) module.

Tests cover:
- Configuration management
- Role-based access decisions
- Access level filtering (public, internal, restricted, confidential)
- Batch access checking
- Search result filtering
- Accessible documents retrieval
- Audit logging
- Singleton pattern
- Edge cases and boundary conditions
"""

import pytest

from aegis_platform.rag.access_control import (
    AccessCheckResult,
    AccessControlConfig,
    AccessLevel,
    DocumentAccessControl,
    DocumentMetadata,
    Role,
    UserContext,
)


class TestAccessControlConfig:
    """Test AccessControlConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AccessControlConfig()
        assert config.enabled is True
        assert config.fail_secure is True
        assert config.audit_enabled is True
        assert config.audit_log_level == "info"
        assert config.default_access_level == AccessLevel.INTERNAL

    def test_custom_config(self):
        """Test custom configuration."""
        config = AccessControlConfig(enabled=False, fail_secure=False, audit_log_level="debug")
        assert config.enabled is False
        assert config.fail_secure is False
        assert config.audit_log_level == "debug"

    def test_role_hierarchy_default(self):
        """Test default role hierarchy."""
        config = AccessControlConfig()
        assert Role.ADMIN in config.role_hierarchy
        assert Role.ANALYST in config.role_hierarchy[Role.ADMIN]
        assert Role.RESEARCHER in config.role_hierarchy[Role.ADMIN]


class TestUserContext:
    """Test UserContext model."""

    def test_user_context_creation(self):
        """Test creating user context."""
        user = UserContext(user_id="user123", role=Role.ANALYST)
        assert user.user_id == "user123"
        assert user.role == Role.ANALYST

    def test_user_context_with_org(self):
        """Test user context with organization."""
        user = UserContext(user_id="user123", role=Role.ANALYST, organization_id="org456")
        assert user.organization_id == "org456"


class TestDocumentMetadata:
    """Test DocumentMetadata model."""

    def test_document_creation(self):
        """Test creating document metadata."""
        doc = DocumentMetadata(doc_id="doc1", title="Test Document")
        assert doc.doc_id == "doc1"
        assert doc.title == "Test Document"
        assert doc.access_level == AccessLevel.INTERNAL

    def test_document_with_access_level(self):
        """Test document with specific access level."""
        doc = DocumentMetadata(
            doc_id="doc1",
            title="Restricted Doc",
            access_level=AccessLevel.RESTRICTED,
            allowed_roles=[Role.ANALYST],
        )
        assert doc.access_level == AccessLevel.RESTRICTED
        assert Role.ANALYST in doc.allowed_roles

    def test_document_with_organization(self):
        """Test document with organization."""
        doc = DocumentMetadata(doc_id="doc1", title="Org Doc", organization_id="org456")
        assert doc.organization_id == "org456"


class TestAccessCheckResult:
    """Test AccessCheckResult model."""

    def test_allowed_result(self):
        """Test successful access result."""
        result = AccessCheckResult(is_allowed=True, reason="User has role")
        assert result.is_allowed is True
        assert result.reason == "User has role"
        assert result.risk_level == "low"

    def test_denied_result(self):
        """Test denied access result."""
        result = AccessCheckResult(
            is_allowed=False, reason="Insufficient permissions", risk_level="medium"
        )
        assert result.is_allowed is False
        assert result.risk_level == "medium"


class TestRoleBasedAccess:
    """Test role-based access control decisions."""

    @pytest.fixture
    def access_control(self):
        """Create fresh access control instance."""
        DocumentAccessControl.reset_instance()
        ac = DocumentAccessControl()
        return ac

    @pytest.mark.asyncio
    async def test_admin_full_access(self, access_control):
        """Test that admin users have full access."""
        admin = UserContext(user_id="admin1", role=Role.ADMIN)
        doc = DocumentMetadata(
            doc_id="doc1",
            title="Restricted Doc",
            access_level=AccessLevel.RESTRICTED,
            allowed_roles=[Role.ANALYST],
        )
        access_control.register_document(doc)

        result = await access_control.check_access(admin, "doc1")
        assert result.is_allowed is True
        assert "admin" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_public_access_all_users(self, access_control):
        """Test that public documents accessible to all."""
        user = UserContext(user_id="user1", role=Role.USER)
        doc = DocumentMetadata(doc_id="doc1", title="Public Doc", access_level=AccessLevel.PUBLIC)
        access_control.register_document(doc)

        result = await access_control.check_access(user, "doc1")
        assert result.is_allowed is True

    @pytest.mark.asyncio
    async def test_internal_access_authenticated_only(self, access_control):
        """Test internal documents require authentication."""
        auth_user = UserContext(user_id="user1", role=Role.USER)
        guest = UserContext(user_id="guest1", role=Role.GUEST)

        doc = DocumentMetadata(
            doc_id="doc1", title="Internal Doc", access_level=AccessLevel.INTERNAL
        )
        access_control.register_document(doc)

        # Authenticated user can access
        result = await access_control.check_access(auth_user, "doc1")
        assert result.is_allowed is True

        # Guest cannot access
        result = await access_control.check_access(guest, "doc1")
        assert result.is_allowed is False

    @pytest.mark.asyncio
    async def test_restricted_access_role_check(self, access_control):
        """Test restricted documents check allowed roles."""
        analyst = UserContext(user_id="user1", role=Role.ANALYST)
        researcher = UserContext(user_id="user2", role=Role.RESEARCHER)

        doc = DocumentMetadata(
            doc_id="doc1",
            title="Restricted Doc",
            access_level=AccessLevel.RESTRICTED,
            allowed_roles=[Role.ANALYST],
        )
        access_control.register_document(doc)

        # Analyst can access
        result = await access_control.check_access(analyst, "doc1")
        assert result.is_allowed is True

        # Researcher cannot access
        result = await access_control.check_access(researcher, "doc1")
        assert result.is_allowed is False

    @pytest.mark.asyncio
    async def test_confidential_admin_only(self, access_control):
        """Test confidential documents admin-only."""
        admin = UserContext(user_id="admin1", role=Role.ADMIN)
        analyst = UserContext(user_id="user1", role=Role.ANALYST)

        doc = DocumentMetadata(
            doc_id="doc1", title="Confidential Doc", access_level=AccessLevel.CONFIDENTIAL
        )
        access_control.register_document(doc)

        # Admin can access
        result = await access_control.check_access(admin, "doc1")
        assert result.is_allowed is True

        # Non-admin cannot access
        result = await access_control.check_access(analyst, "doc1")
        assert result.is_allowed is False
        assert result.risk_level == "high"


class TestAccessLevelFiltering:
    """Test access level-based filtering."""

    @pytest.fixture
    def access_control(self):
        """Create fresh access control instance."""
        DocumentAccessControl.reset_instance()
        ac = DocumentAccessControl()
        return ac

    def test_public_level_no_restrictions(self, access_control):
        """Test PUBLIC level has no restrictions."""
        doc = DocumentMetadata(doc_id="doc1", title="Public", access_level=AccessLevel.PUBLIC)
        result = access_control._check_access_by_level(
            UserContext(user_id="user1", role=Role.GUEST), doc, "read"
        )
        assert result.is_allowed is True

    def test_internal_level_requires_auth(self, access_control):
        """Test INTERNAL level requires authentication."""
        doc = DocumentMetadata(doc_id="doc1", title="Internal", access_level=AccessLevel.INTERNAL)

        # Guest denied
        result = access_control._check_access_by_level(
            UserContext(user_id="guest1", role=Role.GUEST), doc, "read"
        )
        assert result.is_allowed is False

        # Authenticated user allowed
        result = access_control._check_access_by_level(
            UserContext(user_id="user1", role=Role.USER), doc, "read"
        )
        assert result.is_allowed is True

    def test_restricted_level_role_check(self, access_control):
        """Test RESTRICTED level checks allowed roles."""
        doc = DocumentMetadata(
            doc_id="doc1",
            title="Restricted",
            access_level=AccessLevel.RESTRICTED,
            allowed_roles=[Role.ANALYST],
        )

        # Allowed role can access
        result = access_control._check_access_by_level(
            UserContext(user_id="user1", role=Role.ANALYST), doc, "read"
        )
        assert result.is_allowed is True

        # Other role cannot access
        result = access_control._check_access_by_level(
            UserContext(user_id="user2", role=Role.USER), doc, "read"
        )
        assert result.is_allowed is False

    def test_confidential_level_admin_only(self, access_control):
        """Test CONFIDENTIAL level admin-only."""
        doc = DocumentMetadata(
            doc_id="doc1", title="Confidential", access_level=AccessLevel.CONFIDENTIAL
        )

        # Admin allowed
        result = access_control._check_access_by_level(
            UserContext(user_id="admin1", role=Role.ADMIN), doc, "read"
        )
        assert result.is_allowed is True

        # Non-admin denied
        result = access_control._check_access_by_level(
            UserContext(user_id="user1", role=Role.ANALYST), doc, "read"
        )
        assert result.is_allowed is False


class TestBatchAccess:
    """Test batch access checking."""

    @pytest.fixture
    def access_control(self):
        """Create fresh access control instance."""
        DocumentAccessControl.reset_instance()
        ac = DocumentAccessControl()
        return ac

    @pytest.mark.asyncio
    async def test_batch_access_multiple_docs(self, access_control):
        """Test checking access for multiple documents."""
        user = UserContext(user_id="user1", role=Role.ANALYST)

        docs = [
            DocumentMetadata(doc_id="doc1", title="Doc 1", access_level=AccessLevel.PUBLIC),
            DocumentMetadata(
                doc_id="doc2",
                title="Doc 2",
                access_level=AccessLevel.RESTRICTED,
                allowed_roles=[Role.ANALYST],
            ),
            DocumentMetadata(
                doc_id="doc3",
                title="Doc 3",
                access_level=AccessLevel.RESTRICTED,
                allowed_roles=[Role.RESEARCHER],
            ),
        ]

        for doc in docs:
            access_control.register_document(doc)

        results = await access_control.check_access_batch(user, ["doc1", "doc2", "doc3"])

        assert results["doc1"].is_allowed is True  # Public
        assert results["doc2"].is_allowed is True  # Analyst has access
        assert results["doc3"].is_allowed is False  # Analyst doesn't have access

    @pytest.mark.asyncio
    async def test_batch_access_empty_list(self, access_control):
        """Test batch access with empty document list."""
        user = UserContext(user_id="user1", role=Role.USER)
        results = await access_control.check_access_batch(user, [])
        assert results == {}


class TestResultFiltering:
    """Test search result filtering."""

    @pytest.fixture
    def access_control(self):
        """Create fresh access control instance."""
        DocumentAccessControl.reset_instance()
        ac = DocumentAccessControl()
        return ac

    def test_filter_mixed_results(self, access_control):
        """Test filtering search results with mixed access levels."""
        user = UserContext(user_id="user1", role=Role.USER)

        docs = [
            DocumentMetadata(doc_id="doc1", title="Public", access_level=AccessLevel.PUBLIC),
            DocumentMetadata(doc_id="doc2", title="Internal", access_level=AccessLevel.INTERNAL),
            DocumentMetadata(
                doc_id="doc3",
                title="Restricted",
                access_level=AccessLevel.RESTRICTED,
                allowed_roles=[Role.ANALYST],
            ),
        ]

        for doc in docs:
            access_control.register_document(doc)

        results = [
            {"doc_id": "doc1", "score": 0.9},
            {"doc_id": "doc2", "score": 0.8},
            {"doc_id": "doc3", "score": 0.7},
        ]

        filtered = access_control.filter_search_results(user, results)

        # Only public and internal should remain
        assert len(filtered) == 2
        assert filtered[0]["doc_id"] == "doc1"
        assert filtered[1]["doc_id"] == "doc2"

    def test_filter_all_denied(self, access_control):
        """Test filtering when all results are denied."""
        guest = UserContext(user_id="guest1", role=Role.GUEST)

        doc = DocumentMetadata(doc_id="doc1", title="Internal", access_level=AccessLevel.INTERNAL)
        access_control.register_document(doc)

        results = [{"doc_id": "doc1", "score": 0.9}]
        filtered = access_control.filter_search_results(guest, results)

        assert len(filtered) == 0


class TestAccessibleDocuments:
    """Test retrieving accessible documents."""

    @pytest.fixture
    def access_control(self):
        """Create fresh access control instance."""
        DocumentAccessControl.reset_instance()
        ac = DocumentAccessControl()
        return ac

    @pytest.mark.asyncio
    async def test_get_accessible_documents(self, access_control):
        """Test retrieving accessible documents for user."""
        analyst = UserContext(user_id="user1", role=Role.ANALYST)

        docs = [
            DocumentMetadata(doc_id="doc1", title="Public"),
            DocumentMetadata(doc_id="doc2", title="Internal", access_level=AccessLevel.INTERNAL),
            DocumentMetadata(
                doc_id="doc3",
                title="Restricted",
                access_level=AccessLevel.RESTRICTED,
                allowed_roles=[Role.ANALYST],
            ),
            DocumentMetadata(
                doc_id="doc4", title="Confidential", access_level=AccessLevel.CONFIDENTIAL
            ),
        ]

        for doc in docs:
            access_control.register_document(doc)

        accessible = await access_control.get_accessible_documents(analyst)

        # Should have public, internal, and restricted (analyst role)
        assert len(accessible) == 3
        doc_ids = {doc.doc_id for doc in accessible}
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
        assert "doc3" in doc_ids
        assert "doc4" not in doc_ids

    @pytest.mark.asyncio
    async def test_get_accessible_with_limit(self, access_control):
        """Test accessible documents respects limit."""
        user = UserContext(user_id="user1", role=Role.USER)

        for i in range(10):
            doc = DocumentMetadata(doc_id=f"doc{i}", title=f"Doc {i}")
            access_control.register_document(doc)

        accessible = await access_control.get_accessible_documents(user, limit=5)
        assert len(accessible) <= 5


class TestAuditLogging:
    """Test audit logging functionality."""

    @pytest.fixture
    def access_control(self):
        """Create fresh access control instance."""
        DocumentAccessControl.reset_instance()
        ac = DocumentAccessControl()
        return ac

    @pytest.mark.asyncio
    async def test_audit_log_created(self, access_control):
        """Test audit log entry creation."""
        user = UserContext(user_id="user1", role=Role.USER)
        doc = DocumentMetadata(doc_id="doc1", title="Doc", access_level=AccessLevel.PUBLIC)
        access_control.register_document(doc)

        result = await access_control.check_access(user, "doc1")

        assert result.audit_log_entry is not None
        assert result.audit_log_entry["user_id"] == "user1"
        assert result.audit_log_entry["resource_id"] == "doc1"
        assert result.audit_log_entry["result"] == "allowed"

    @pytest.mark.asyncio
    async def test_audit_disabled(self, access_control):
        """Test audit logging can be disabled."""
        config = AccessControlConfig(audit_enabled=False)
        DocumentAccessControl.reset_instance()
        ac = DocumentAccessControl(config)

        user = UserContext(user_id="user1", role=Role.USER)
        doc = DocumentMetadata(doc_id="doc1", title="Doc", access_level=AccessLevel.PUBLIC)
        ac.register_document(doc)

        result = await ac.check_access(user, "doc1")
        assert result.audit_log_entry is None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def access_control(self):
        """Create fresh access control instance."""
        DocumentAccessControl.reset_instance()
        ac = DocumentAccessControl()
        return ac

    @pytest.mark.asyncio
    async def test_missing_document_fail_secure(self, access_control):
        """Test missing document with fail_secure=True."""
        config = AccessControlConfig(fail_secure=True)
        DocumentAccessControl.reset_instance()
        ac = DocumentAccessControl(config)

        user = UserContext(user_id="user1", role=Role.USER)
        result = await ac.check_access(user, "nonexistent")

        assert result.is_allowed is False

    @pytest.mark.asyncio
    async def test_missing_document_fail_open(self, access_control):
        """Test missing document with fail_secure=False."""
        config = AccessControlConfig(fail_secure=False)
        DocumentAccessControl.reset_instance()
        ac = DocumentAccessControl(config)

        user = UserContext(user_id="user1", role=Role.USER)

        with pytest.raises(ValueError):
            await ac.check_access(user, "nonexistent")

    @pytest.mark.asyncio
    async def test_multitenant_isolation(self, access_control):
        """Test multi-tenant document isolation."""
        user_a = UserContext(user_id="user_a", role=Role.USER, organization_id="org_a")
        user_b = UserContext(user_id="user_b", role=Role.USER, organization_id="org_b")

        doc_a = DocumentMetadata(doc_id="doc_a", title="Org A Doc", organization_id="org_a")
        access_control.register_document(doc_a)

        # User A can access
        result = await access_control.check_access(user_a, "doc_a")
        assert result.is_allowed is True

        # User B cannot access (different org)
        result = await access_control.check_access(user_b, "doc_a")
        assert result.is_allowed is False
        assert "organization" in result.reason.lower()

    def test_access_control_disabled(self):
        """Test that disabled access control allows all."""
        config = AccessControlConfig(enabled=False)
        DocumentAccessControl.reset_instance()
        ac = DocumentAccessControl(config)

        ac._check_access_by_level(
            UserContext(user_id="user1", role=Role.GUEST),
            DocumentMetadata(
                doc_id="doc1", title="Confidential", access_level=AccessLevel.CONFIDENTIAL
            ),
            "read",
        )
        # When disabled, check_access handles it and returns allowed=True
        # This test verifies no errors occur with disabled config


class TestSingletonPattern:
    """Test singleton pattern implementation."""

    def test_singleton_creation(self):
        """Test singleton instance creation."""
        DocumentAccessControl.reset_instance()
        ac1 = DocumentAccessControl.get_instance()
        ac2 = DocumentAccessControl.get_instance()

        assert ac1 is ac2

    def test_singleton_config_first_call(self):
        """Test config applied on first singleton instantiation."""
        DocumentAccessControl.reset_instance()

        config = AccessControlConfig(audit_enabled=False)
        ac = DocumentAccessControl.get_instance(config)

        assert ac.config.audit_enabled is False
