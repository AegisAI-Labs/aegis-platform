"""Single error classification for the agent API (ADR-011).

Both ``/agent/run`` (HTTP status + JSON body) and ``/agent/stream`` (SSE ``error`` event) use
``classify_error`` so the same failure is always reported the same way. Messages are fixed strings:
provider exception text is never returned to clients.
"""

import logging
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from aegis_platform.llm.errors import (
    LLMProviderError,
    ProviderAuthenticationError,
    ProviderQuotaExhausted,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)
from aegis_platform.observability.context import get_correlation_id
from aegis_platform.tools.errors import ToolExecutionError

logger = logging.getLogger(__name__)

PROVIDER_FAILED_MESSAGE = "The AI provider could not complete the request."
QUOTA_EXHAUSTED_MESSAGE = "Provider quota exhausted. Please add credits or switch providers."


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    status_code: int
    code: str
    message: str


def classify_error(exc: BaseException) -> ErrorInfo:
    if isinstance(exc, ToolExecutionError):
        return ErrorInfo(400, "invalid_tool_input", str(exc)[:300])
    if isinstance(exc, ProviderRateLimited):
        return ErrorInfo(429, "provider_rate_limited", PROVIDER_FAILED_MESSAGE)
    if isinstance(exc, ProviderQuotaExhausted):
        return ErrorInfo(503, "provider_quota_exhausted", QUOTA_EXHAUSTED_MESSAGE)
    if isinstance(exc, ProviderUnavailable):
        return ErrorInfo(503, "provider_unavailable", PROVIDER_FAILED_MESSAGE)
    if isinstance(exc, ProviderTimeout | TimeoutError):
        return ErrorInfo(504, "provider_timeout", PROVIDER_FAILED_MESSAGE)
    if isinstance(exc, ProviderAuthenticationError):
        return ErrorInfo(502, "provider_authentication_failed", PROVIDER_FAILED_MESSAGE)
    if isinstance(exc, LLMProviderError):
        return ErrorInfo(502, "provider_error", PROVIDER_FAILED_MESSAGE)
    return ErrorInfo(500, "internal_error", "An unexpected error occurred.")


def error_body(info: ErrorInfo) -> dict[str, dict[str, str]]:
    return {
        "error": {
            "code": info.code,
            "message": info.message,
            "correlation_id": get_correlation_id(),
        }
    }


async def _handle_known_error(request: Request, exc: Exception) -> JSONResponse:
    info = classify_error(exc)
    logger.warning(
        "agent_request_failed",
        extra={"code": info.code, "error_type": type(exc).__name__, "path": request.url.path},
    )
    return JSONResponse(status_code=info.status_code, content=error_body(info))


def register_exception_handlers(app: FastAPI) -> None:
    """Map tool and provider failures to structured responses.

    Unexpected exceptions keep FastAPI's default 500 handling.
    """
    for exc_type in (ToolExecutionError, LLMProviderError, TimeoutError):
        app.add_exception_handler(exc_type, _handle_known_error)
