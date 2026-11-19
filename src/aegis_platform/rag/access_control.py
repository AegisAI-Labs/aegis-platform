"""Document Access Control and Role-Based Access Control (RBAC) for RAG.

This module implements role-based document access filtering for RAG systems,
ensuring users can only access documents they have permission to view.

Features:
- Role-based access control (RBAC) with configurable role hierarchy
- Document access levels (public, internal, restricted, confidential)
- Single-document and batch access checking
- Search result filtering by user permissions
- Audit logging for compliance tracking
- Singleton pattern for thread-safe operation
- Multi-tenant support ready

Example:
    >>> from aegis_platform.rag.access_control import (
    ...     DocumentAccessControl, UserContext, DocumentMetadata,
    ...     Role, AccessLevel, AccessControlConfig
    ... )
    >>>
    >>> # Create user context
    >>> user = UserContext(user_id="user123", role=Role.ANALYST)
    >>>
    >>> # Create document with metadata
    >>> doc = DocumentMetadata(
    ...     doc_id="doc1",
    ...     title="Financial Report",
    ...     access_level=AccessLevel.RESTRICTED,
    ...     allowed_roles=[Role.ANALYST, Role.ADMIN]
    ... )
    >>>
    >>> # Check access
    >>> ac = DocumentAccessControl.get_instance()
    >>> result = await ac.check_access(user, "doc1")
    >>> if result.is_allowed:
    ...     print("User can access document")
"""

import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """Available roles in the system."""

    ADMIN = "admin"  # Full access to all documents
    ANALYST = "analyst"  # Access to business intelligence docs
    RESEARCHER = "researcher"  # Access to research papers and data
    USER = "user"  # Access to public documents
    GUEST = "guest"  # No access


class AccessLevel(str, Enum):
    """Document access levels."""

    PUBLIC = "public"  # Anyone can access
    INTERNAL = "internal"  # Authenticated users only
    RESTRICTED = "restricted"  # Specific roles only
    CONFIDENTIAL = "confidential"  # Admin only


class DocumentMetadata(BaseModel):
    """Document metadata for access control."""

    model_config = ConfigDict(use_enum_values=False)

    doc_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    access_level: AccessLevel = Field(
        default=AccessLevel.INTERNAL, description="Document access level"
    )
    allowed_roles: list[Role] = Field(
        default_factory=list, description="Roles that can access this document"
    )
    owner: str | None = Field(default=None, description="User ID who owns/uploaded the document")
    organization_id: str | None = Field(
        default=None, description="Organization ID for multi-tenant support"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Document creation timestamp"
    )
    tags: list[str] = Field(
        default_factory=list, description="Tags for filtering (domain, subject, etc)"
    )


class UserContext(BaseModel):
    """User context for access checking."""

    model_config = ConfigDict(use_enum_values=False)

    user_id: str = Field(..., description="Unique user identifier")
    role: Role = Field(..., description="User's role")
    organization_id: str | None = Field(
        default=None, description="Organization ID for multi-tenant support"
    )
    permissions: list[str] = Field(
        default_factory=list, description="Custom permissions beyond role"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="User context creation timestamp"
    )


class AccessCheckResult(BaseModel):
    """Result of access control check."""

    model_config = ConfigDict(use_enum_values=False)

    is_allowed: bool = Field(..., description="Whether access is allowed")
    reason: str = Field(..., description="Why access was allowed/denied")
    risk_level: str = Field(
        default="low", description="Risk level of this access attempt ('low', 'medium', 'high')"
    )
    audit_log_entry: dict | None = Field(
        default=None, description="Audit log entry for this access check"
    )


class AccessControlConfig(BaseModel):
    """Access control configuration."""

    model_config = ConfigDict(use_enum_values=False)

    enabled: bool = Field(default=True, description="Enable access control checks")
    fail_secure: bool = Field(
        default=True, description="Return empty on permission error (not throw error)"
    )
    audit_enabled: bool = Field(default=True, description="Enable audit logging")
    audit_log_level: str = Field(
        default="info", description="Audit log level (debug, info, warning, error)"
    )
    role_hierarchy: dict[Role, list[Role]] = Field(
        default_factory=lambda: {
            Role.ADMIN: [Role.ANALYST, Role.RESEARCHER, Role.USER, Role.GUEST],
            Role.ANALYST: [Role.USER, Role.GUEST],
            Role.RESEARCHER: [Role.USER, Role.GUEST],
            Role.USER: [Role.GUEST],
            Role.GUEST: [],
        },
        description="Role inheritance hierarchy (parent role inherits child permissions)",
    )
    default_access_level: AccessLevel = Field(
        default=AccessLevel.INTERNAL,
        description="Default access level for documents without explicit level",
    )


class DocumentAccessControl:
    """Document access control engine using role-based access control (RBAC).

    Implements a singleton pattern for thread-safe operation. Provides methods
    for checking individual and batch document access, filtering search results,
    and logging audit trails.
    """

    _instance: Optional["DocumentAccessControl"] = None

    def __init__(self, config: AccessControlConfig | None = None):
        """Initialize access control with configuration.

        Args:
            config: AccessControlConfig instance. If None, uses defaults.
        """
        self.config = config or AccessControlConfig()
        self.document_registry: dict[str, DocumentMetadata] = {}

    async def check_access(
        self, user: UserContext, doc_id: str, operation: str = "read"
    ) -> AccessCheckResult:
        """Check if user can access a specific document.

        Args:
            user: User context (ID, role, permissions)
            doc_id: Document ID to check access for
            operation: Operation type (read, write, delete). Default is "read".

        Returns:
            AccessCheckResult with is_allowed flag and reason

        Raises:
            ValueError: If document not found in registry and fail_secure is False
        """
        if not self.config.enabled:
            logger.debug(
                "Access control disabled, allowing access",
                extra={"user_id": user.user_id, "doc_id": doc_id},
            )
            return AccessCheckResult(
                is_allowed=True, reason="Access control disabled", risk_level="low"
            )

        # Check if document exists in registry
        if doc_id not in self.document_registry:
            if self.config.fail_secure:
                logger.warning(
                    "Document not found, failing secure",
                    extra={"doc_id": doc_id, "user_id": user.user_id},
                )
                result = AccessCheckResult(
                    is_allowed=False, reason="Document not found", risk_level="low"
                )
                await self._log_access_attempt(user, doc_id, operation, result)
                return result
            raise ValueError(f"Document {doc_id} not found in registry")

        doc = self.document_registry[doc_id]

        result = self.evaluate(user, doc, operation)

        # Log the access attempt
        await self._log_access_attempt(user, doc_id, operation, result)

        return result

    def evaluate(
        self, user: UserContext, doc: DocumentMetadata, operation: str = "read"
    ) -> AccessCheckResult:
        """Decide access for a document without consulting the registry.

        Applies the multi-tenant check, then the access-level rules. Use this when
        document metadata travels with the data (e.g. vector store payloads).

        Args:
            user: User context
            doc: Document metadata
            operation: Operation type (read, write, delete)

        Returns:
            AccessCheckResult with access decision (not audit-logged)
        """
        if not self.config.enabled:
            return AccessCheckResult(
                is_allowed=True, reason="Access control disabled", risk_level="low"
            )

        if (
            user.organization_id
            and doc.organization_id
            and user.organization_id != doc.organization_id
        ):
            return AccessCheckResult(
                is_allowed=False,
                reason="User organization does not have access to this document",
                risk_level="medium",
            )

        return self._check_access_by_level(user, doc, operation)

    async def check_access_batch(
        self, user: UserContext, doc_ids: list[str]
    ) -> dict[str, AccessCheckResult]:
        """Check access for multiple documents (for RAG search results).

        Args:
            user: User context
            doc_ids: List of document IDs to check

        Returns:
            Dictionary mapping doc_id to AccessCheckResult
        """
        results = {}
        for doc_id in doc_ids:
            results[doc_id] = await self.check_access(user, doc_id)
        return results

    def filter_search_results(self, user: UserContext, search_results: list[dict]) -> list[dict]:
        """Filter search results by user access permissions.

        Synchronous version for use in search pipeline. Returns only documents
        the user has permission to access.

        Args:
            user: User context
            search_results: List of search result dictionaries with 'doc_id' field

        Returns:
            Filtered list containing only accessible documents
        """
        filtered = []

        for result in search_results:
            doc_id = result.get("doc_id") or result.get("id")

            if doc_id not in self.document_registry:
                # Skip unknown documents
                if not self.config.fail_secure:
                    filtered.append(result)
                continue

            doc = self.document_registry[doc_id]

            if self.evaluate(user, doc, "read").is_allowed:
                filtered.append(result)

        logger.info(
            "Search results filtered by access control",
            extra={
                "user_id": user.user_id,
                "original_count": len(search_results),
                "filtered_count": len(filtered),
            },
        )

        return filtered

    async def get_accessible_documents(
        self, user: UserContext, limit: int = 100
    ) -> list[DocumentMetadata]:
        """Get all documents the user can access.

        Args:
            user: User context
            limit: Maximum number of documents to return

        Returns:
            List of accessible DocumentMetadata objects
        """
        accessible: list[DocumentMetadata] = []

        for doc in self.document_registry.values():
            if len(accessible) >= limit:
                break

            if self.evaluate(user, doc, "read").is_allowed:
                accessible.append(doc)

        logger.info(
            "Retrieved accessible documents for user",
            extra={"user_id": user.user_id, "count": len(accessible), "limit": limit},
        )

        return accessible

    def register_document(self, doc: DocumentMetadata) -> None:
        """Register a document in the access control registry.

        Args:
            doc: DocumentMetadata to register
        """
        self.document_registry[doc.doc_id] = doc
        logger.debug(
            "Document registered",
            extra={"doc_id": doc.doc_id, "access_level": doc.access_level.value},
        )

    def _check_access_by_level(
        self, user: UserContext, doc: DocumentMetadata, operation: str
    ) -> AccessCheckResult:
        """Check access based on document access level.

        Args:
            user: User context
            doc: Document metadata
            operation: Operation type (read, write, delete)

        Returns:
            AccessCheckResult with access decision
        """
        # Admin always has access
        if user.role == Role.ADMIN:
            return AccessCheckResult(
                is_allowed=True, reason="Admin user has full access", risk_level="low"
            )

        # Check public access
        if doc.access_level == AccessLevel.PUBLIC:
            return AccessCheckResult(is_allowed=True, reason="Document is public", risk_level="low")

        # For other levels, user must be authenticated
        if user.role == Role.GUEST:
            return AccessCheckResult(
                is_allowed=False, reason="Guest users cannot access this document", risk_level="low"
            )

        # Check INTERNAL access (any authenticated user)
        if doc.access_level == AccessLevel.INTERNAL:
            return AccessCheckResult(
                is_allowed=True,
                reason="User is authenticated and document is internal",
                risk_level="low",
            )

        # Check RESTRICTED access (specific roles)
        if doc.access_level == AccessLevel.RESTRICTED:
            if user.role in doc.allowed_roles:
                return AccessCheckResult(
                    is_allowed=True,
                    reason=f"User role {user.role.value} in allowed roles",
                    risk_level="low",
                )

            # Check role hierarchy (inherited permissions)
            if self._check_role_hierarchy(user.role, doc.allowed_roles):
                return AccessCheckResult(
                    is_allowed=True,
                    reason=f"User role {user.role.value} inherits from allowed role",
                    risk_level="low",
                )

            return AccessCheckResult(
                is_allowed=False,
                reason=f"User role {user.role.value} not in allowed roles",
                risk_level="medium",
            )

        # Check CONFIDENTIAL access (admin only)
        if doc.access_level == AccessLevel.CONFIDENTIAL:
            return AccessCheckResult(
                is_allowed=False,
                reason="Document is confidential, admin access required",
                risk_level="high",
            )

        # Default deny
        return AccessCheckResult(
            is_allowed=False, reason="Access denied by default", risk_level="low"
        )

    def _check_role_hierarchy(self, user_role: Role, allowed_roles: list[Role]) -> bool:
        """Check if user role inherits from allowed roles via hierarchy.

        Args:
            user_role: User's role
            allowed_roles: List of roles that have access

        Returns:
            True if user role inherits from any allowed role
        """
        # Get inherited roles from hierarchy
        inherited = self.config.role_hierarchy.get(user_role, [])

        # Check if any allowed role is in inherited roles
        for allowed_role in allowed_roles:
            if allowed_role in inherited:
                return True

        return False

    async def _log_access_attempt(
        self, user: UserContext, doc_id: str, operation: str, result: AccessCheckResult
    ) -> None:
        """Log access attempt for audit trail.

        Args:
            user: User context
            doc_id: Document ID
            operation: Operation type
            result: Access check result
        """
        if not self.config.audit_enabled:
            return

        audit_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "user_id": user.user_id,
            "user_role": user.role.value,
            "organization_id": user.organization_id,
            "operation": operation,
            "resource_type": "document",
            "resource_id": doc_id,
            "action": "access_check",
            "result": "allowed" if result.is_allowed else "denied",
            "reason": result.reason,
            "risk_level": result.risk_level,
        }

        result.audit_log_entry = audit_entry

        log_level = getattr(logging, self.config.audit_log_level.upper(), logging.INFO)
        logger.log(
            log_level,
            "Access control audit log",
            extra={
                "audit_entry": audit_entry,
                "user_id": user.user_id,
                "resource_id": doc_id,
                "result": result.is_allowed,
            },
        )

    @staticmethod
    def get_instance(config: AccessControlConfig | None = None) -> "DocumentAccessControl":
        """Get singleton instance of DocumentAccessControl.

        Args:
            config: Configuration for first instantiation. Ignored if instance
                   already created. Future calls should use get_instance() without args.

        Returns:
            Singleton DocumentAccessControl instance
        """
        if DocumentAccessControl._instance is None:
            DocumentAccessControl._instance = DocumentAccessControl(config)
        return DocumentAccessControl._instance

    @staticmethod
    def reset_instance() -> None:
        """Reset singleton instance (useful for testing)."""
        DocumentAccessControl._instance = None
