"""X-Correlation-ID middleware: accept safe IDs, generate otherwise, echo, and reset."""

import json
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from aegis_platform.observability.context import (
    CORRELATION_ID_HEADER,
    _correlation_id,
    get_correlation_id,
    is_valid_correlation_id,
)
from aegis_platform.observability.middleware import CorrelationIdMiddleware


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/id")
    def current_id() -> dict[str, str]:
        return {"correlation_id": get_correlation_id()}

    @app.get("/stream")
    def stream() -> StreamingResponse:
        async def body():  # type: ignore[no-untyped-def]
            yield json.dumps({"correlation_id": get_correlation_id()})

        return StreamingResponse(body(), media_type="text/event-stream")

    return TestClient(app)


def test_incoming_id_is_used_and_echoed(client: TestClient) -> None:
    response = client.get("/id", headers={CORRELATION_ID_HEADER: "req-abc_123.x"})

    assert response.json() == {"correlation_id": "req-abc_123.x"}
    assert response.headers[CORRELATION_ID_HEADER] == "req-abc_123.x"


def test_missing_id_is_generated_and_echoed(client: TestClient) -> None:
    response = client.get("/id")

    generated = response.headers[CORRELATION_ID_HEADER]
    UUID(generated)  # valid UUID
    assert response.json() == {"correlation_id": generated}


@pytest.mark.parametrize(
    "unsafe",
    ["has space", "a" * 129, "line\\nbreak", "<script>", ""],
)
def test_unsafe_incoming_id_is_replaced(client: TestClient, unsafe: str) -> None:
    response = client.get("/id", headers={CORRELATION_ID_HEADER: unsafe})

    echoed = response.headers[CORRELATION_ID_HEADER]
    assert echoed != unsafe
    UUID(echoed)


def test_each_request_gets_its_own_id(client: TestClient) -> None:
    first = client.get("/id").headers[CORRELATION_ID_HEADER]
    second = client.get("/id").headers[CORRELATION_ID_HEADER]

    assert first != second


def test_id_is_bound_while_streaming(client: TestClient) -> None:
    response = client.get("/stream", headers={CORRELATION_ID_HEADER: "stream-1"})

    assert response.headers[CORRELATION_ID_HEADER] == "stream-1"
    assert json.loads(response.text) == {"correlation_id": "stream-1"}


def test_id_is_reset_after_request(client: TestClient) -> None:
    client.get("/id", headers={CORRELATION_ID_HEADER: "transient"})

    assert _correlation_id.get() != "transient"


@pytest.mark.parametrize(
    ("value", "valid"),
    [("abc", True), ("A-b_c.1", True), ("x" * 128, True), ("x" * 129, False), ("", False)],
)
def test_is_valid_correlation_id(value: str, valid: bool) -> None:
    assert is_valid_correlation_id(value) is valid


def test_platform_app_echoes_correlation_id(client: TestClient) -> None:
    from importlib import import_module
    from unittest.mock import patch

    from aegis_platform.llm.factory import LLMProviderFactory
    from tests.fakes import StubLLMProvider

    with patch.object(LLMProviderFactory, "create_llm_provider", return_value=StubLLMProvider()):
        api = import_module("aegis_platform.api.main")

    response = TestClient(api.app).get("/health", headers={CORRELATION_ID_HEADER: "health-1"})

    assert response.headers[CORRELATION_ID_HEADER] == "health-1"
