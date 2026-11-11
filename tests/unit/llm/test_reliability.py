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


class FlakyGenerateProvider:
    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        if self.generate_calls == 1:
            raise TimeoutError("temporary generation failure")
        return LLMResponse(content="recovered", model="test", provider="test")

    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        async def chunks() -> AsyncIterator[str]:
            yield "unused"

        return chunks()


@pytest.mark.asyncio
async def test_generate_notifies_callback_before_retry() -> None:
    provider = FlakyGenerateProvider()
    retry_events: list[tuple[int, float, Exception]] = []
    request = LLMRequest(messages=[LLMMessage(role="user", content="Hello")])
    retrying_provider = RetryingLLMProvider(
        provider,
        policy=RetryPolicy(
            max_attempts=2,
            timeout_seconds=1,
            initial_delay_seconds=0,
            max_delay_seconds=0,
            jitter=False,
        ),
        sleep=lambda _: _no_sleep(),
        on_retry=lambda attempt, delay, error: retry_events.append((attempt, delay, error)),
    )

    response = await retrying_provider.generate(request)

    assert response.content == "recovered"
    assert provider.generate_calls == 2
    assert len(retry_events) == 1
    attempt, delay, error = retry_events[0]
    assert attempt == 1
    assert delay == 0
    assert isinstance(error, TimeoutError)
    assert str(error) == "temporary generation failure"


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
