"""ASGI middleware that binds a correlation ID to every HTTP request.

Pure ASGI (not ``BaseHTTPMiddleware``) so the ID stays bound while streaming responses
are produced, and so the ID is echoed on every response, including SSE.
"""

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .context import (
    CORRELATION_ID_HEADER,
    bind_correlation_id,
    is_valid_correlation_id,
    new_correlation_id,
    reset_correlation_id,
)


class CorrelationIdMiddleware:
    def __init__(self, app: ASGIApp, header_name: str = CORRELATION_ID_HEADER) -> None:
        self.app = app
        self.header_name = header_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(self.header_name)
        correlation_id = incoming if is_valid_correlation_id(incoming) else new_correlation_id()

        async def send_with_header(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[self.header_name] = correlation_id
            await send(message)

        token = bind_correlation_id(correlation_id)
        try:
            await self.app(scope, receive, send_with_header)
        finally:
            reset_correlation_id(token)
