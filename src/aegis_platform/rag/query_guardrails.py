"""Query Guardrails for RAG Platform.

This module provides security validation and sanitization for RAG queries,
protecting against:
- SQL injection attacks
- Prompt injection attacks
- Shell command injection attacks
- Query length exploitation

SQL checks match statement shapes (``DROP TABLE``, ``UNION SELECT``, ``' OR '1'='1``)
rather than bare keywords, so ordinary questions are not blocked.
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
        block_command_patterns: Whether to block shell command injection patterns
    """

    enabled: bool = True
    max_query_length: int = 10000
    max_query_tokens: int = 2000
    block_sql_patterns: bool = True
    block_prompt_patterns: bool = True
    block_command_patterns: bool = True


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

    # SQL injection: statement shapes rather than bare keywords, so natural-language
    # queries like "How do I select a model?" pass. (keyword, pattern) pairs.
    SQL_INJECTION_PATTERNS: ClassVar[list[tuple[str, str]]] = [
        ("UNION", r"\bunion\s+(?:all\s+)?select\b"),
        ("SELECT", r"\bselect\s+(?:\*|distinct\b|count\s*\(|[\w.]+\s*,\s*[\w.]+\s+from\b)"),
        ("SELECT", r"\bselect\b.+\bfrom\b.+\bwhere\b"),
        ("INSERT", r"\binsert\s+into\b"),
        ("UPDATE", r"\bupdate\s+[\w.]+\s+set\b"),
        ("DELETE", r"\bdelete\s+from\b"),
        ("DROP", r"\bdrop\s+(?:table|database|schema|view|index|user)\b"),
        ("TRUNCATE", r"\btruncate\s+table\b"),
        ("ALTER", r"\balter\s+(?:table|database|user)\b"),
        ("CREATE", r"\bcreate\s+(?:table|database|user|procedure|function)\s+[\w.]+\s*(?:\(|;|$)"),
        ("EXEC", r"\bexec(?:ute)?\s*(?:\(|xp_|sp_)|\bxp_cmdshell\b"),
        ("INFORMATION_SCHEMA", r"\binformation_schema\b"),
        ("SLEEP", r"\bwaitfor\s+delay\b|\bsleep\s*\(\s*\d|\bbenchmark\s*\("),
        ("CHAR", r"\bchar\s*\(\s*\d"),
    ]

    # Comment markers only count when they terminate an injected string or statement.
    SQL_COMMENT_PATTERN: ClassVar[str] = r"['\";)]\s*(?:--|#)|/\*.*?\*/"

    # Classic tautologies: ' OR '1'='1, ' OR 1=1, OR 1=1
    SQL_TAUTOLOGY_PATTERN: ClassVar[str] = (
        r"'\s*(?:or|and)\s+'?\w+'?\s*(?:=|like)\s*'?\w+|\bor\s+(\d+)\s*=\s*\1\b"
    )

    # Shell command injection: a separator or substitution followed by a command,
    # destructive flags, and sensitive system paths.
    SHELL_COMMANDS: ClassVar[str] = (
        r"rm|curl|wget|chmod|chown|sh|bash|zsh|nc|ncat|netcat|python\d*|perl|ruby|php|"
        r"powershell|pwsh|cmd|del|shutdown|reboot|kill|cat|mkfifo|dd|scp|ssh|eval|exec"
    )
    COMMAND_INJECTION_PATTERNS: ClassVar[list[tuple[str, str]]] = [
        (
            "shell command after separator",
            rf"(?:;|&&|\|\||\||`|\$\()\s*(?:sudo\s+)?(?:{SHELL_COMMANDS})\b",
        ),
        ("command substitution", r"\$\(\s*\w+[^)]*\)"),
        ("destructive rm flags", r"\brm\s+-[a-z]*[rf]"),
        ("sensitive system file", r"/etc/(?:passwd|shadow|sudoers)\b"),
        ("path traversal", r"(?:\.\./){2,}"),
        ("network redirect", r">\s*/dev/(?:tcp|udp)/"),
    ]

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

        if self.config.block_command_patterns:
            command_violations = self._check_command_injection(query)
            if command_violations:
                violations.extend(command_violations)
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
        violations: list[str] = []
        flags = re.IGNORECASE | re.DOTALL

        detected: list[str] = []
        for keyword, pattern in self.SQL_INJECTION_PATTERNS:
            if keyword not in detected and re.search(pattern, query, flags):
                detected.append(keyword)
        violations.extend(f"Detected SQL keyword: {keyword}" for keyword in detected)

        if re.search(self.SQL_COMMENT_PATTERN, query, flags):
            violations.append("Detected SQL comment patterns")

        if re.search(self.SQL_TAUTOLOGY_PATTERN, query, flags):
            violations.append("Detected SQL tautology pattern")

        return violations

    def _check_command_injection(self, query: str) -> list[str]:
        """Check for shell command injection patterns.

        Args:
            query: Query text to check

        Returns:
            List of detected command injection violations
        """
        return [
            f"Detected command injection pattern: {name}"
            for name, pattern in self.COMMAND_INJECTION_PATTERNS
            if re.search(pattern, query, re.IGNORECASE)
        ]

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
            sanitized = re.sub(
                r"--|/\*.*?\*/|(?<=['\";])\s*#.*?$", " ", sanitized, flags=re.MULTILINE
            )

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
