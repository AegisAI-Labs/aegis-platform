from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx2 import Request, Response
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)

from aegis_platform.llm.errors import (
    ProviderAuthenticationError,
    ProviderRateLimited,
    ProviderRequestError,
    ProviderTimeout,
    ProviderUnavailable,
)
from aegis_platform.llm.models import LLMMessage, LLMRequest, LLMResponse, LLMUsage
from aegis_platform.llm.providers.nvidia import NvidiaProvider


@pytest.mark.asyncio
async def test_nvidia_generate_converts_messages_and_usage() -> None:
    provider = NvidiaProvider(api_key="test-key", model="default-model", timeout_seconds=10)
    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content="Be brief"),
            LLMMessage(role="developer", content="Be helpful"),
            LLMMessage(role="user", content="Hi"),
            LLMMessage(role="assistant", content="Hello"),
        ],
        model="override-model",
        temperature=0.2,
        max_tokens=20,
    )
    sdk_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello there"))],
        model="override-model",
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6),
    )

    with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = sdk_response
        result = await provider.generate(request)

    assert result == LLMResponse(
        content="Hello there",
        model="override-model",
        provider="nvidia",
        usage=LLMUsage(input_tokens=4, output_tokens=2, total_tokens=6),
    )
    assert create.call_args.kwargs["model"] == "override-model"
    assert create.call_args.kwargs["messages"] == [
        {"role": "system", "content": "Be brief"},
        {"role": "developer", "content": "Be helpful"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    assert create.call_args.kwargs["temperature"] == 0.2
    assert create.call_args.kwargs["max_tokens"] == 20
    assert create.call_args.kwargs["stream"] is False


@pytest.mark.asyncio
async def test_nvidia_generate_rejects_empty_choices() -> None:
    provider = NvidiaProvider(api_key="test-key", model="default-model", timeout_seconds=10)
    request = LLMRequest(messages=[LLMMessage(role="user", content="Hi")])

    with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = SimpleNamespace(choices=[], model="default-model", usage=None)
        with pytest.raises(ProviderRequestError, match="NVIDIA returned no completion choices"):
            await provider.generate(request)


@pytest.mark.asyncio
async def test_nvidia_stream_yields_nonempty_chunks() -> None:
    provider = NvidiaProvider(api_key="test-key", model="default-model", timeout_seconds=10)
    request = LLMRequest(messages=[LLMMessage(role="user", content="Hi")])

    async def chunks():
        yield SimpleNamespace(choices=[])
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))])
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])

    with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = chunks()
        result = [chunk async for chunk in provider.stream(request)]

    assert result == ["Hi"]
    assert create.call_args.kwargs["model"] == "default-model"
    assert create.call_args.kwargs["stream"] is True


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("unavailable", ProviderUnavailable),
        ("timeout", ProviderTimeout),
        ("rate_limit", ProviderRateLimited),
        ("authentication", ProviderAuthenticationError),
        ("other", ProviderRequestError),
    ],
)
@pytest.mark.asyncio
async def test_nvidia_generate_maps_sdk_errors(
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
        "other": OpenAIError("unexpected SDK error"),
    }
    provider = NvidiaProvider(api_key="test-key", model="default-model", timeout_seconds=10)
    llm_request = LLMRequest(messages=[LLMMessage(role="user", content="Hi")])

    with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.side_effect = sdk_errors[failure]
        with pytest.raises(expected_error) as raised:
            await provider.generate(llm_request)

    assert raised.value.__cause__ is sdk_errors[failure]
