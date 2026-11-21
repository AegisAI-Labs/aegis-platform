"""NVIDIA embedding provider against a mocked HTTP transport (no network)."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
import pytest

from aegis_platform.rag.embeddings.base import EmbeddingProvider
from aegis_platform.rag.embeddings.nvidia import NVIDIAEmbeddingConfig, NVIDIAEmbeddingProvider
from aegis_platform.rag.models import EmbeddingRequest, EmbeddingUsage

DIM = 4


def make_provider(handler, **config) -> tuple[NVIDIAEmbeddingProvider, list[dict]]:
    """Provider whose HTTP client is served by ``handler``; returns captured JSON payloads."""
    payloads: list[dict] = []

    def capture(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return handler(request, payloads[-1])

    provider = NVIDIAEmbeddingProvider(
        NVIDIAEmbeddingConfig(
            api_key="nvapi-test", embedding_dimension=DIM, max_retries=2, **config
        )
    )
    provider.client = httpx.AsyncClient(transport=httpx.MockTransport(capture))
    return provider, payloads


def ok(_: httpx.Request, payload: dict) -> httpx.Response:
    data = [{"index": i, "embedding": [float(i)] * DIM} for i in range(len(payload["input"]))]
    return httpx.Response(200, json={"data": list(reversed(data))})


@pytest.fixture(autouse=True)
def no_backoff_sleep():
    with patch(
        "aegis_platform.rag.embeddings.nvidia.asyncio.sleep", new_callable=AsyncMock
    ) as sleep:
        yield sleep


def test_conforms_to_embedding_provider() -> None:
    provider, _ = make_provider(ok)
    assert isinstance(provider, EmbeddingProvider)
    assert provider.model == "nvidia/nemotron-3.5-embed-1b"


@pytest.mark.asyncio
async def test_queries_are_sent_as_query_input_type() -> None:
    provider, payloads = make_provider(ok)

    vector = await provider.embed_text("how often do keys rotate?")

    assert payloads[0]["input_type"] == "query"
    assert isinstance(vector, np.ndarray) and vector.dtype == np.float32


@pytest.mark.asyncio
async def test_documents_are_sent_as_passage_input_type_and_realigned() -> None:
    provider, payloads = make_provider(ok)

    vectors = await provider.embed_batch(["a", "b", "c"])

    assert payloads[0]["input_type"] == "passage"
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]  # sorted by index


@pytest.mark.asyncio
async def test_input_type_can_be_disabled_for_symmetric_models() -> None:
    provider, payloads = make_provider(ok, send_input_type=False)

    await provider.embed_text("hello")

    assert "input_type" not in payloads[0]


@pytest.mark.asyncio
async def test_batches_are_split() -> None:
    provider, payloads = make_provider(ok, batch_size=2)

    vectors = await provider.embed_batch(["a", "b", "c", "d", "e"])

    assert len(vectors) == 5
    assert [len(p["input"]) for p in payloads] == [2, 2, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 403, 404, 422])
async def test_client_errors_fail_fast_without_body(status: int) -> None:
    def reject(_: httpx.Request, __: dict) -> httpx.Response:
        return httpx.Response(status, json={"message": "secret echoed input"})

    provider, payloads = make_provider(reject)

    with pytest.raises(RuntimeError, match=f"HTTP {status}") as raised:
        await provider.embed_text("hello")

    assert len(payloads) == 1  # no retries
    assert "secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_server_errors_are_retried_then_succeed(no_backoff_sleep) -> None:
    responses = iter([httpx.Response(503, json={"message": "Service is overloaded"})])

    def flaky(request: httpx.Request, payload: dict) -> httpx.Response:
        return next(responses, None) or ok(request, payload)

    provider, payloads = make_provider(flaky)

    vector = await provider.embed_text("hello")

    assert len(payloads) == 2
    assert vector.shape == (DIM,)


@pytest.mark.asyncio
async def test_retries_exhausted_reports_status_only() -> None:
    def down(_: httpx.Request, __: dict) -> httpx.Response:
        return httpx.Response(503, json={"message": "overloaded"})

    provider, payloads = make_provider(down)

    with pytest.raises(RuntimeError, match=r"after 3 attempts \(HTTP 503\)") as raised:
        await provider.embed_text("hello")

    assert len(payloads) == 3
    assert "integrate.api.nvidia.com" not in str(raised.value)


@pytest.mark.asyncio
async def test_wrong_dimension_is_rejected() -> None:
    def wrong(_: httpx.Request, __: dict) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * (DIM + 1)}]})

    provider, _ = make_provider(wrong)

    with pytest.raises(ValueError, match="Unexpected embedding dimension"):
        await provider.embed_text("hello")


def test_from_env_reads_send_input_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_KEY", "nvapi-test")
    monkeypatch.setenv("RAG_EMBEDDING_SEND_INPUT_TYPE", "false")

    assert NVIDIAEmbeddingConfig.from_env().send_input_type is False


def with_usage(tokens: int | None, model: str = "nvidia/served-model"):
    def handler(request: httpx.Request, payload: dict) -> httpx.Response:
        body = ok(request, payload).json()
        body["model"] = model
        if tokens is not None:
            body["usage"] = {"prompt_tokens": tokens, "total_tokens": tokens}
        return httpx.Response(200, json=body)

    return handler


@pytest.mark.asyncio
async def test_embed_returns_usage_model_and_provider() -> None:
    provider, payloads = make_provider(with_usage(6), batch_size=2)

    response = await provider.embed(EmbeddingRequest(texts=["a", "b", "c"], input_type="query"))

    assert response.usage == EmbeddingUsage(input_tokens=12)  # 2 API calls x 6
    assert response.model == "nvidia/served-model"
    assert response.provider == "nvidia"
    assert response.dimension == DIM
    assert len(response.embeddings) == 3
    assert {p["input_type"] for p in payloads} == {"query"}


@pytest.mark.asyncio
async def test_embed_without_usage_reports_none() -> None:
    provider, _ = make_provider(with_usage(None))

    response = await provider.embed(EmbeddingRequest(texts=["a"]))

    assert response.usage is None


@pytest.mark.asyncio
async def test_embed_rejects_empty_texts() -> None:
    provider, payloads = make_provider(ok)

    with pytest.raises(ValueError, match=r"positions \[1\]"):
        await provider.embed(EmbeddingRequest(texts=["a", " "]))

    assert payloads == []
