# ContextVar is suitable for async request context. It avoids using a global variable shared by concurrent requests.
# At the API boundary, we can later accept or generate a correlation ID and set it for the duration of the request.

from contextvars import ContextVar
from uuid import uuid4

_correlation_id: ContextVar[str | None] = ContextVar(
    "aegis_correlation_id",
    default=None,
)


def get_correlation_id() -> str:
    current = _correlation_id.get()

    if current:
        return current

    generated = str(uuid4())
    _correlation_id.set(generated)

    return generated


def set_correlation_id(value: str) -> None:
    if not value or not value.strip():
        raise ValueError("Correlation ID cannot be empty")

    _correlation_id.set(value)


def clear_correlation_id() -> None:
    _correlation_id.set(None)
