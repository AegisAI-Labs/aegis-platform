"""Real streaming through the LangGraph workflow (ADR-011)."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from aegis_platform.agent.service import AgentService
from aegis_platform.llm.errors import ProviderTimeout
from aegis_platform.llm.models import LLMRequest
from aegis_platform.observability import InMemoryEventSink, InMemoryMetricsSink, ObservedLLMProvider
from tests.fakes import ChunkedStubLLMProvider, FailingStubLLMProvider, StubLLMProvider


@pytest.mark.asyncio
async def test_stream_forwards_provider_chunks_unchanged() -> None:
    provider = ChunkedStubLLMProvider(("Hel", "lo", " world"))

    chunks = [chunk async for chunk in AgentService(provider).stream("Hello")]

    assert chunks == ["Hel", "lo", " world"]
    assert len(provider.requests) == 1  # streamed once, generate() not used


@pytest.mark.asyncio
async def test_stream_calculator_yields_result_once() -> None:
    provider = StubLLMProvider()

    chunks = [chunk async for chunk in AgentService(provider).stream("calculate 7*9")]

    assert chunks == ["63"]
    assert provider.requests == []


@pytest.mark.asyncio
async def test_stream_first_chunk_arrives_before_generation_finishes() -> None:
    release = asyncio.Event()

    class SlowTail(StubLLMProvider):
        async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
            yield "first"
            await release.wait()
            yield "second"

    stream = AgentService(SlowTail()).stream("Hello")

    first = await asyncio.wait_for(anext(stream), timeout=1)
    assert first == "first"  # delivered while the provider is still generating

    release.set()
    assert [chunk async for chunk in stream] == ["second"]


@pytest.mark.asyncio
async def test_stream_error_propagates_after_partial_output() -> None:
    provider = FailingStubLLMProvider(ProviderTimeout("raw"), chunks_before_error=("partial",))
    received: list[str] = []

    with pytest.raises(ProviderTimeout):
        async for chunk in AgentService(provider).stream("Hello"):
            received.append(chunk)

    assert received == ["partial"]


@pytest.mark.asyncio
async def test_stream_cancellation_reaches_provider_and_telemetry() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class Hanging(StubLLMProvider):
        async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
            yield "first"
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            yield "never"

    metrics = InMemoryMetricsSink()
    events = InMemoryEventSink()
    observed = ObservedLLMProvider(Hanging(), metrics, events, default_model="m")

    async def consume() -> None:
        async for _ in AgentService(observed).stream("Hello"):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()
    names = [p.name for p in metrics.snapshot()]
    assert names.count("llm_requests_cancelled_total") == 1
    assert "llm_requests_failed_total" not in names


@pytest.mark.asyncio
async def test_run_still_uses_generate() -> None:
    provider = ChunkedStubLLMProvider(("a", "b"))

    result = await AgentService(provider).run("Hello")

    assert result == {"response": "ab", "tool_used": "llm"}
