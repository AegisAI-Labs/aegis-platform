"""Query Guardrails for RAG Platform.

This module provides security validation and sanitization for RAG queries,
protecting against:
- SQL injection attacks
- Prompt injection attacks
- Malicious query patterns
- Query length exploitation

The guardrails enforce a whitelist-based approach, validating that queries
conform to expected patterns before processing.
"""

import logging
import re
from dataclasses import dataclass
from typing import ClassVar

logger = logging.getLogger(__name__)


@dataclass
class GuardrailsConfig:
    """Configuration for query guardrails.

    Attributes:
        enabled: Whether guardrails are enabled
        max_query_length: Maximum allowed query length (default: 10000)
        max_query_tokens: Maximum allowed query tokens (approximate)
        block_sql_patterns: Whether to block SQL injection patterns
        block_prompt_patterns: Whether to block prompt injection patterns
        allow_special_chars: Set of allowed special characters
    """

    enabled: bool = True
    max_query_length: int = 10000
    max_query_tokens: int = 2000
    block_sql_patterns: bool = True
    block_prompt_patterns: bool = True
    allow_special_chars: str = "!?.,;:'-\"()"


class GuardResult:
    """Result of guardrail validation.

    Attributes:
        is_safe: Whether query passed all checks
        query: Original or sanitized query
        violations: List of detected violations
        blocked: Whether query was blocked
        risk_level: Risk level (LOW, MEDIUM, HIGH)
    """

    def __init__(
        self,
        is_safe: bool,
        query: str,
        violations: list[str] | None = None,
        risk_level: str = "LOW",
    ):
        self.is_safe = is_safe
        self.query = query
        self.violations = violations or []
        self.blocked = not is_safe
        self.risk_level = risk_level

    def __repr__(self) -> str:
        return (
            f"GuardResult(safe={self.is_safe}, risk={self.risk_level}, "
            f"violations={len(self.violations)}, blocked={self.blocked})"
        )


class QueryGuardrails:
    """Security guardrails for RAG query validation.

    Protects against:
    - SQL injection patterns (UNION, DROP, DELETE, etc.)
    - Prompt injection patterns (system prompts, instruction overrides)
    - Length-based DoS attacks
    - Malformed queries

    Uses whitelist-based approach: rejects patterns that look suspicious
    rather than trying to fix them.
    """

    # SQL injection keywords to detect
    SQL_INJECTION_KEYWORDS: ClassVar[set[str]] = {
        "UNION",
        "SELECT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "TRUNCATE",
        "EXEC",
        "EXECUTE",
        "SCRIPT",
        "DECLARE",
        "CAST",
        "CONVERT",
        "INFORMATION_SCHEMA",
        "xp_",
        "sp_",
        "0x",
        "char(",
        "ascii(",
        "substring(",
    }

    # Prompt injection patterns
    PROMPT_INJECTION_PATTERNS: ClassVar[list[str]] = [
        r"(?i)system\s*:\s*",  # system: prompts
        r"(?i)assistant\s*:\s*",  # assistant: role overrides
        r"(?i)human\s*:\s*",  # human: role overrides
        r"(?i)ignore.*previous",  # ignore previous instructions
        r"(?i)forget.*everything",  # forget everything
        r"(?i)pretend.*you.*are",  # pretend you are
        r"\[SYSTEM\]",
        r"\[ADMIN\]",  # bracketed system commands
        r"<!--.*-->",  # HTML comments
        r"\{\{.*\}\}",  # template injection
        r"\$\{.*\}",  # template variables
    ]

    def __init__(self, config: GuardrailsConfig | None = None):
        """Initialize Query Guardrails.

        Args:
            config: GuardrailsConfig object (uses defaults if None)
        """
        self.config = config or GuardrailsConfig()
        logger.info("QueryGuardrails initialized with config: %s", self.config)

    def validate_query(self, query: str) -> GuardResult:
        """Validate query against all guardrails.

        Args:
            query: Query text to validate

        Returns:
            GuardResult with validation status and violations

        Raises:
            ValueError: If query is None or empty
        """
        if not query:
            raise ValueError("Query cannot be empty")

        if not self.config.enabled:
            logger.debug("Guardrails disabled, allowing query")
            return GuardResult(is_safe=True, query=query)

        violations = []
        risk_level = "LOW"

        # Check length
        if len(query) > self.config.max_query_length:
            violations.append(
                f"Query exceeds maximum length ({len(query)} > {self.config.max_query_length})"
            )
            risk_level = "HIGH"

        # Check token count (approximate: words * 1.3)
        estimated_tokens = int(len(query.split()) * 1.3)
        if estimated_tokens > self.config.max_query_tokens:
            violations.append(
                f"Query exceeds maximum tokens ({estimated_tokens} > {self.config.max_query_tokens})"
            )
            risk_level = "HIGH"

        # Check SQL injection patterns
        if self.config.block_sql_patterns:
            sql_violations = self._check_sql_injection(query)
            if sql_violations:
                violations.extend(sql_violations)
                # SQL injection is HIGH risk (not MEDIUM)
                risk_level = "HIGH"

        # Check prompt injection patterns
        if self.config.block_prompt_patterns:
            prompt_violations = self._check_prompt_injection(query)
            if prompt_violations:
                violations.extend(prompt_violations)
                # Prompt injection is HIGH risk (not MEDIUM)
                risk_level = "HIGH"

        # Check for null bytes and control characters
        if "\x00" in query or any(ord(c) < 32 and c not in "\t\n\r" for c in query):
            violations.append("Query contains control characters")
            risk_level = "HIGH"

        is_safe = len(violations) == 0

        if not is_safe:
            logger.warning(
                "Query validation failed: %s violations=%s",
                risk_level,
                violations,
            )
        else:
            logger.debug("Query validation passed")

        return GuardResult(
            is_safe=is_safe, query=query, violations=violations, risk_level=risk_level
        )

    def _check_sql_injection(self, query: str) -> list[str]:
        """Check for SQL injection patterns.

        Args:
            query: Query text to check

        Returns:
            List of detected SQL injection violations
        """
        violations = []
        query_upper = query.upper()

        for keyword in self.SQL_INJECTION_KEYWORDS:
            # Look for SQL keywords with word boundaries or special patterns
            if keyword.endswith("_"):
                # For patterns like xp_, sp_, check for exact match
                if keyword in query_upper:
                    violations.append(f"Detected SQL pattern: {keyword}")
            else:
                # For keywords, use word boundaries
                pattern = rf"\b{re.escape(keyword)}\b"
                if re.search(pattern, query_upper):
                    violations.append(f"Detected SQL keyword: {keyword}")

        # Check for SQL comment patterns
        if re.search(r"--|/\*|\*/|#", query):
            violations.append("Detected SQL comment patterns")

        # Check for escape sequences
        if re.search(r"'.*'.*'", query):  # Multiple quotes
            violations.append("Detected suspicious quote patterns")

        return violations

    def _check_prompt_injection(self, query: str) -> list[str]:
        """Check for prompt injection patterns.

        Args:
            query: Query text to check

        Returns:
            List of detected prompt injection violations
        """
        violations = []

        for pattern in self.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, query):
                violations.append(f"Detected prompt injection pattern: {pattern}")

        return violations

    def sanitize_query(self, query: str) -> str:
        """Sanitize query by removing/escaping dangerous content.

        WARNING: This is a best-effort approach. Use validate_query()
        to check if query is safe before processing.

        Args:
            query: Query text to sanitize

        Returns:
            Sanitized query text
        """
        if not query:
            return query

        sanitized = query

        # Remove null bytes
        sanitized = sanitized.replace("\x00", "")

        # Remove control characters (but keep tab, newline, carriage return)
        sanitized = "".join(c for c in sanitized if ord(c) >= 32 or c in "\t\n\r")

        # Remove SQL comments if blocking SQL patterns
        if self.config.block_sql_patterns:
            sanitized = re.sub(r"--|/\*.*?\*/|#.*?$", " ", sanitized, flags=re.MULTILINE)

        # Normalize whitespace
        sanitized = " ".join(sanitized.split())

        # Truncate if too long
        if len(sanitized) > self.config.max_query_length:
            sanitized = sanitized[: self.config.max_query_length]

        logger.debug("Query sanitized: %d -> %d chars", len(query), len(sanitized))

        return sanitized


# Singleton instance for use throughout the application
_query_guardrails: QueryGuardrails | None = None


def get_query_guardrails(config: GuardrailsConfig | None = None) -> QueryGuardrails:
    """Get or create singleton QueryGuardrails instance.

    Args:
        config: GuardrailsConfig object (only used on first call)

    Returns:
        QueryGuardrails instance
    """
    global _query_guardrails
    if _query_guardrails is None:
        _query_guardrails = QueryGuardrails(config)
    return _query_guardrails
