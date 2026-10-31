from collections.abc import AsyncIterator

import pytest

from aegis_platform.llm.models import LLMMessage, LLMRequest, LLMResponse
from aegis_platform.llm.reliability import RetryingLLMProvider, RetryPolicy


class FlakyStreamProvider:
    def __init__(self, *, fail_after_chunk: bool) -> None:
        self.fail_after_chunk = fail_after_chunk
        self.stream_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        self.stream_calls += 1
        if self.fail_after_chunk:
            yield "partial"
        elif self.stream_calls > 1:
            yield "complete"
        if self.fail_after_chunk or self.stream_calls == 1:
            raise TimeoutError("temporary stream failure")


def make_retrying_provider(provider: FlakyStreamProvider) -> RetryingLLMProvider:
    return RetryingLLMProvider(
        provider,
        policy=RetryPolicy(
            max_attempts=2,
            timeout_seconds=1,
            initial_delay_seconds=0,
            max_delay_seconds=0,
            jitter=False,
        ),
        sleep=lambda _: _no_sleep(),
    )


async def _no_sleep() -> None:
    return None


@pytest.mark.asyncio
async def test_stream_retries_timeout_before_yielding_output() -> None:
    provider = FlakyStreamProvider(fail_after_chunk=False)
    retrying_provider = make_retrying_provider(provider)
    request = LLMRequest(messages=[LLMMessage(role="user", content="Hello")])

    chunks = [chunk async for chunk in retrying_provider.stream(request)]

    assert chunks == ["complete"]
    assert provider.stream_calls == 2


@pytest.mark.asyncio
async def test_stream_does_not_retry_after_yielding_output() -> None:
    provider = FlakyStreamProvider(fail_after_chunk=True)
    retrying_provider = make_retrying_provider(provider)
    request = LLMRequest(messages=[LLMMessage(role="user", content="Hello")])
    chunks: list[str] = []

    with pytest.raises(TimeoutError, match="temporary stream failure"):
        async for chunk in retrying_provider.stream(request):
            chunks.append(chunk)

    assert chunks == ["partial"]
    assert provider.stream_calls == 1
