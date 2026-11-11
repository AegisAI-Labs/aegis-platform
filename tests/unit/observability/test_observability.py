from dataclasses import dataclass, field

import pytest

from aegis_platform.observability import (
    InMemoryEventSink,
    InMemoryMetricsSink,
    ObservedLLMProvider,
)


@dataclass
class FakeUsage:
    input_tokens: int = 12
    output_tokens: int = 7


@dataclass
class FakeResponse:
    content: str = "hello"
    model: str = "fake-model"
    provider: str = "fake"
    usage: FakeUsage = field(default_factory=FakeUsage)


@dataclass
class FakeRequest:
    model: str = "fake-model"


class FakeProvider:
    provider_name = "fake"

    async def generate(self, request):
        return FakeResponse()

    async def stream(self, request):
        yield "hello"
        yield " world"


class FailingProvider:
    provider_name = "fake"

    async def generate(self, request):
        raise TimeoutError("sensitive raw message")

    async def stream(self, request):
        yield "partial"

        raise TimeoutError("sensitive raw message")


@pytest.mark.asyncio
async def test_generate_records_request_and_usage():
    metrics = InMemoryMetricsSink()
    events = InMemoryEventSink()

    provider = ObservedLLMProvider(
        FakeProvider(),
        metrics,
        events,
    )

    response = await provider.generate(FakeRequest())

    assert response.content == "hello"

    names = [point.name for point in metrics.snapshot()]

    assert "llm_requests_total" in names
    assert "llm_request_duration_seconds" in names
    assert "llm_input_tokens_total" in names
    assert "llm_output_tokens_total" in names

    assert events.snapshot()[0].name == "llm.request.completed"


@pytest.mark.asyncio
async def test_stream_records_ttft_and_duration():
    metrics = InMemoryMetricsSink()
    events = InMemoryEventSink()

    provider = ObservedLLMProvider(
        FakeProvider(),
        metrics,
        events,
    )

    chunks = [chunk async for chunk in provider.stream(FakeRequest())]

    assert chunks == ["hello", " world"]

    names = [point.name for point in metrics.snapshot()]

    assert "llm_time_to_first_token_seconds" in names

    assert "llm_request_duration_seconds" in names

    assert events.snapshot()[0].name == "llm.stream.completed"


@pytest.mark.asyncio
async def test_generate_failure_is_recorded():
    metrics = InMemoryMetricsSink()
    events = InMemoryEventSink()

    provider = ObservedLLMProvider(
        FailingProvider(),
        metrics,
        events,
    )

    with pytest.raises(TimeoutError):
        await provider.generate(FakeRequest())

    names = [point.name for point in metrics.snapshot()]

    assert "llm_requests_failed_total" in names

    event_text = repr(events.snapshot())

    assert "TimeoutError" in event_text
    assert "sensitive raw message" not in event_text


@pytest.mark.asyncio
async def test_stream_interruption_is_recorded():
    metrics = InMemoryMetricsSink()
    events = InMemoryEventSink()

    provider = ObservedLLMProvider(
        FailingProvider(),
        metrics,
        events,
    )

    chunks = []

    with pytest.raises(TimeoutError):
        async for chunk in provider.stream(FakeRequest()):
            chunks.append(chunk)

    assert chunks == ["partial"]

    names = [point.name for point in metrics.snapshot()]

    assert "llm_stream_interruptions_total" in names

    assert "llm_requests_failed_total" in names

    assert events.snapshot()[0].name == "llm.stream.interrupted"
