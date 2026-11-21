"""HTTP and SSE error mapping for the agent API (ADR-011)."""

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from aegis_platform.agent.service import AgentService
from aegis_platform.llm.errors import (
    LLMProviderError,
    ProviderAuthenticationError,
    ProviderQuotaExhausted,
    ProviderRateLimited,
    ProviderRequestError,
    ProviderTimeout,
    ProviderUnavailable,
)
from tests.fakes import FailingStubLLMProvider

PROVIDER_ERRORS = [
    (ProviderRateLimited("raw secret"), 429, "provider_rate_limited"),
    (ProviderQuotaExhausted("raw secret"), 503, "provider_quota_exhausted"),
    (ProviderUnavailable("raw secret"), 503, "provider_unavailable"),
    (ProviderTimeout("raw secret"), 504, "provider_timeout"),
    (TimeoutError("raw secret"), 504, "provider_timeout"),
    (ProviderAuthenticationError("raw secret"), 502, "provider_authentication_failed"),
    (ProviderRequestError("raw secret"), 502, "provider_error"),
    (LLMProviderError("raw secret"), 502, "provider_error"),
]


@pytest.fixture
def use_provider(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    import aegis_platform.api.main as api

    def install(provider: object) -> None:
        monkeypatch.setattr(api, "agent_service", AgentService(provider))  # type: ignore[arg-type]

    yield install


def sse_events(text: str) -> list[tuple[str, dict[str, str]]]:
    events = []
    for block in text.strip().split("\n\n"):
        event_line, data_line = block.split("\n")
        events.append(
            (event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: ")))
        )
    return events


@pytest.mark.parametrize(("error", "status", "code"), PROVIDER_ERRORS)
def test_run_maps_provider_errors(
    client: TestClient, use_provider, error: Exception, status: int, code: str
) -> None:
    use_provider(FailingStubLLMProvider(error))

    response = client.post(
        "/agent/run", json={"message": "Hello"}, headers={"X-Correlation-ID": "c-1"}
    )

    assert response.status_code == status
    body = response.json()
    assert body["error"]["code"] == code
    assert body["error"]["correlation_id"] == "c-1"
    assert "raw secret" not in response.text


@pytest.mark.parametrize(("error", "status", "code"), PROVIDER_ERRORS)
def test_stream_maps_provider_errors_to_error_event(
    client: TestClient, use_provider, error: Exception, status: int, code: str
) -> None:
    use_provider(FailingStubLLMProvider(error))

    with client.stream("POST", "/agent/stream", json={"message": "Hello"}) as response:
        text = "".join(response.iter_text())

    assert response.status_code == 200
    [(event, data)] = sse_events(text)
    assert event == "error"
    assert data["code"] == code
    assert "raw secret" not in text


def test_stream_error_after_partial_output(client: TestClient, use_provider) -> None:
    use_provider(
        FailingStubLLMProvider(ProviderUnavailable("x"), chunks_before_error=("Hi", " there"))
    )

    with client.stream("POST", "/agent/stream", json={"message": "Hello"}) as response:
        events = sse_events("".join(response.iter_text()))

    assert events == [
        ("token", {"text": "Hi"}),
        ("token", {"text": " there"}),
        (
            "error",
            {
                "message": "The AI provider could not complete the request.",
                "code": "provider_unavailable",
            },
        ),
    ]


def test_stream_keeps_quota_message(client: TestClient, use_provider) -> None:
    use_provider(FailingStubLLMProvider(ProviderQuotaExhausted("x")))

    with client.stream("POST", "/agent/stream", json={"message": "Hello"}) as response:
        [(_, data)] = sse_events("".join(response.iter_text()))

    assert data["message"] == "Provider quota exhausted. Please add credits or switch providers."


def test_stream_unexpected_error_still_ends_with_error_event(
    client: TestClient, use_provider
) -> None:
    use_provider(FailingStubLLMProvider(RuntimeError("internal detail")))

    with client.stream("POST", "/agent/stream", json={"message": "Hello"}) as response:
        text = "".join(response.iter_text())

    [(event, data)] = sse_events(text)
    assert event == "error"
    assert data == {"message": "An unexpected error occurred.", "code": "internal_error"}
    assert "internal detail" not in text


def test_stream_invalid_calculation_reports_tool_error(client: TestClient) -> None:
    with client.stream("POST", "/agent/stream", json={"message": "calculate 1/0"}) as response:
        [(event, data)] = sse_events("".join(response.iter_text()))

    assert event == "error"
    assert data == {"message": "Division by zero", "code": "invalid_tool_input"}


def test_stream_calculator_success(client: TestClient) -> None:
    with client.stream("POST", "/agent/stream", json={"message": "calculate 7*9"}) as response:
        events = sse_events("".join(response.iter_text()))

    assert events == [("token", {"text": "63"}), ("done", {"status": "completed"})]
