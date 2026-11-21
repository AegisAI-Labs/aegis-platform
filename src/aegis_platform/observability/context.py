"""Request-scoped correlation ID.

A ContextVar keeps the ID isolated per async task, so concurrent requests never share it.
The HTTP boundary binds an ID per request (see ``middleware.py``) and resets it afterwards.
"""

import re
from contextvars import ContextVar, Token
from typing import TypeGuard
from uuid import uuid4

CORRELATION_ID_HEADER = "X-Correlation-ID"

# Accept only short, log-safe identifiers from callers to prevent log/header injection.
_SAFE_CORRELATION_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

_correlation_id: ContextVar[str | None] = ContextVar(
    "aegis_correlation_id",
    default=None,
)


def new_correlation_id() -> str:
    return str(uuid4())


def is_valid_correlation_id(value: str | None) -> TypeGuard[str]:
    return value is not None and _SAFE_CORRELATION_ID.fullmatch(value) is not None


def get_correlation_id() -> str:
    """Return the current correlation ID, creating one for this context if none is bound."""
    current = _correlation_id.get()

    if current:
        return current

    generated = new_correlation_id()
    _correlation_id.set(generated)

    return generated


def set_correlation_id(value: str) -> None:
    if not value or not value.strip():
        raise ValueError("Correlation ID cannot be empty")

    _correlation_id.set(value)


def bind_correlation_id(value: str) -> Token[str | None]:
    """Bind ``value`` for the current context; pass the token to ``reset_correlation_id``."""
    if not value or not value.strip():
        raise ValueError("Correlation ID cannot be empty")

    return _correlation_id.set(value)


def reset_correlation_id(token: Token[str | None]) -> None:
    _correlation_id.reset(token)


def clear_correlation_id() -> None:
    _correlation_id.set(None)
