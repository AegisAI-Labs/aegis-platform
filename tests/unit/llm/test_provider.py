from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx2 import Request, Response
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

from aegis_platform.llm.base import LLMProvider
from aegis_platform.llm.errors import (
    ProviderAuthenticationError,
    ProviderRateLimited,
    ProviderRequestError,
    ProviderTimeout,
    ProviderUnavailable,
)
from aegis_platform.llm.models import LLMMessage, LLMRequest, LLMResponse, LLMUsage
from aegis_platform.llm.providers.openai import OpenAIProvider
from tests.fakes import StubLLMProvider


@pytest.mark.asyncio
async def test_fake_provider_contract() -> None:
    provider: LLMProvider = StubLLMProvider()
    request = LLMRequest(messages=[LLMMessage(role="user", content="Hello")])

    response = await provider.generate(request)
    chunks = [chunk async for chunk in provider.stream(request)]

    assert isinstance(response, LLMResponse)
    assert response.content == "Hello from the stub LLM"
    assert chunks == [response.content]


@pytest.mark.asyncio
async def test_openai_generate_converts_messages_and_usage() -> None:
    provider = OpenAIProvider(api_key="test-key", model="default-model")
    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content="Be brief"),
            LLMMessage(role="user", content="Hi"),
        ],
        model="override-model",
    )
    sdk_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello"))],
        model="override-model",
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
    )
    with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = sdk_response
        result = await provider.generate(request)

    assert result == LLMResponse(
        content="Hello",
        model="override-model",
        provider="openai",
        usage=LLMUsage(input_tokens=2, output_tokens=3, total_tokens=5),
    )
    assert create.call_args.kwargs["messages"] == [
        {"role": "system", "content": "Be brief"},
        {"role": "user", "content": "Hi"},
    ]
    assert create.call_args.kwargs["stream"] is False


@pytest.mark.asyncio
async def test_openai_stream_yields_nonempty_chunks() -> None:
    provider = OpenAIProvider(api_key="test-key", model="default-model")
    request = LLMRequest(messages=[LLMMessage(role="user", content="Hi")])

    async def chunks():
        yield SimpleNamespace(choices=[])
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))])
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])

    with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = chunks()
        result = [chunk async for chunk in provider.stream(request)]

    assert result == ["Hi"]
    assert create.call_args.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_openai_generate_maps_sdk_errors() -> None:
    provider = OpenAIProvider(api_key="test-key", model="default-model")
    request = LLMRequest(messages=[LLMMessage(role="user", content="Hi")])
    with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.side_effect = RuntimeError("offline")
        with pytest.raises(ProviderRequestError, match="OpenAI request failed: offline"):
            await provider.generate(request)


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("unavailable", ProviderUnavailable),
        ("timeout", ProviderTimeout),
        ("rate_limit", ProviderRateLimited),
        ("authentication", ProviderAuthenticationError),
    ],
)
@pytest.mark.asyncio
async def test_openai_generate_maps_provider_failures(
    failure: str, expected_error: type[Exception]
) -> None:
    request = Request("POST", "https://example.test/chat/completions")
    sdk_errors = {
        "unavailable": APIConnectionError(request=request),
        "timeout": APITimeoutError(request=request),
        "rate_limit": RateLimitError(
            "rate limited", response=Response(429, request=request), body=None
        ),
        "authentication": AuthenticationError(
            "unauthorized", response=Response(401, request=request), body=None
        ),
    }
    provider = OpenAIProvider(api_key="test-key", model="default-model")
    llm_request = LLMRequest(messages=[LLMMessage(role="user", content="Hi")])
    with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.side_effect = sdk_errors[failure]
        with pytest.raises(expected_error) as raised:
            await provider.generate(llm_request)

    assert raised.value.__cause__ is sdk_errors[failure]


@pytest.mark.asyncio
async def test_openai_stream_maps_timeout() -> None:
    provider = OpenAIProvider(api_key="test-key", model="default-model")
    request = LLMRequest(messages=[LLMMessage(role="user", content="Hi")])
    timeout = APITimeoutError(request=Request("POST", "https://example.test/chat/completions"))
    with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.side_effect = timeout
        with pytest.raises(ProviderTimeout) as raised:
            _ = [chunk async for chunk in provider.stream(request)]

    assert raised.value.__cause__ is timeout
