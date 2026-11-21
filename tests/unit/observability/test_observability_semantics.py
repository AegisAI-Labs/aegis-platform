"""Telemetry semantics for ObservedLLMProvider (ADR-008, master context §7.5).

Each test maps to one required case:
 1 success metrics            6 stream completion        11 missing token usage
 2 failure metrics            7 stream interruption      12 metrics sink failure isolation
 3 request duration           8 cancellation             13 event sink failure isolation
 4 retry counter + event      9 early stream closure     14 correlation ID propagation
 5 time to first token       10 empty stream             15 no duplicate logical counts
"""

import asyncio
from collections import Counter
from collections.abc import AsyncIterator

import pytest

from aegis_platform.llm.models import LLMRequest, LLMResponse, LLMUsage
from aegis_platform.llm.reliability import RetryingLLMProvider, RetryPolicy
from aegis_platform.observability import (
    InMemoryEventSink,
    InMemoryMetricsSink,
    MetricPoint,
    ObservabilityEvent,
    ObservedLLMProvider,
)
from aegis_platform.observability.context import (
    bind_correlation_id,
    reset_correlation_id,
)
from aegis_platform.observability.retry_telemetry import RetryTelemetry

REQUEST = LLMRequest(messages=[])


class ScriptedProvider:
    """Provider whose generate/stream behavior is scripted per test."""

    provider_name = "fake"

    def __init__(
        self,
        *,
        response: LLMResponse | None = None,
        generate_error: BaseException | None = None,
        chunks: tuple[str, ...] = ("hello", " world"),
        stream_error_after: int | None = None,
        stream_error: BaseException | None = None,
        stream_delay: float = 0.0,
    ) -> None:
        self.response = response or LLMResponse(
            content="hello",
            model="fake-model",
            provider="fake",
            usage=LLMUsage(input_tokens=12, output_tokens=7),
        )
        self.generate_error = generate_error
        self.chunks = chunks
        self.stream_error_after = stream_error_after
        self.stream_error = stream_error
        self.stream_delay = stream_delay
        self.generate_calls = 0
        self.stream_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        if self.stream_delay:
            await asyncio.sleep(self.stream_delay)
        if self.generate_error is not None:
            raise self.generate_error
        return self.response

    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        async def iterate() -> AsyncIterator[str]:
            self.stream_calls += 1
            for index, chunk in enumerate(self.chunks):
                if self.stream_error_after is not None and index == self.stream_error_after:
                    assert self.stream_error is not None
                    raise self.stream_error
                if self.stream_delay:
                    await asyncio.sleep(self.stream_delay)
                yield chunk
            if self.stream_error_after == len(self.chunks) and self.stream_error:
                raise self.stream_error

        return iterate()


class BrokenMetricsSink:
    def record(self, point: MetricPoint) -> None:
        raise RuntimeError("metrics backend down")


class BrokenEventSink:
    def emit(self, event: ObservabilityEvent) -> None:
        raise RuntimeError("event backend down")


def observe(
    provider: object,
    *,
    default_model: str | None = "configured-model",
) -> tuple[ObservedLLMProvider, InMemoryMetricsSink, InMemoryEventSink]:
    metrics = InMemoryMetricsSink()
    events = InMemoryEventSink()
    observed = ObservedLLMProvider(
        provider,  # type: ignore[arg-type]
        metrics,
        events,
        default_model=default_model,
    )
    return observed, metrics, events


def counts(metrics: InMemoryMetricsSink) -> Counter[str]:
    return Counter(point.name for point in metrics.snapshot())


def event_names(events: InMemoryEventSink) -> list[str]:
    return [event.name for event in events.snapshot()]


# 1 -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_successful_generation_metrics() -> None:
    observed, metrics, events = observe(ScriptedProvider())

    response = await observed.generate(REQUEST)

    assert response.content == "hello"
    assert counts(metrics) == Counter(
        {
            "llm_requests_total": 1,
            "llm_request_duration_seconds": 1,
            "llm_input_tokens_total": 1,
            "llm_output_tokens_total": 1,
        }
    )
    tokens = {p.name: p.value for p in metrics.snapshot() if p.name.endswith("tokens_total")}
    assert tokens == {"llm_input_tokens_total": 12.0, "llm_output_tokens_total": 7.0}
    assert event_names(events) == ["llm.request.completed"]


# 2 -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_failed_generation_metrics_and_no_sensitive_text() -> None:
    observed, metrics, events = observe(
        ScriptedProvider(generate_error=TimeoutError("secret prompt text sk-123"))
    )

    with pytest.raises(TimeoutError):
        await observed.generate(REQUEST)

    assert counts(metrics) == Counter(
        {
            "llm_requests_total": 1,
            "llm_request_duration_seconds": 1,
            "llm_requests_failed_total": 1,
        }
    )
    [event] = events.snapshot()
    assert event.name == "llm.request.failed"
    assert event.attributes["error_type"] == "TimeoutError"
    assert "secret" not in repr(event) and "sk-123" not in repr(event)


# 3 -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_request_duration_measures_whole_operation() -> None:
    observed, metrics, _ = observe(ScriptedProvider(stream_delay=0.05))

    await observed.generate(REQUEST)

    [duration] = [p for p in metrics.snapshot() if p.name == "llm_request_duration_seconds"]
    assert duration.value >= 0.05


# 4 -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_retry_counter_and_event_with_single_logical_request() -> None:
    metrics = InMemoryMetricsSink()
    events = InMemoryEventSink()

    class FlakyOnce(ScriptedProvider):
        async def generate(self, request: LLMRequest) -> LLMResponse:
            self.generate_calls += 1
            if self.generate_calls == 1:
                raise TimeoutError("transient")
            return self.response

    async def no_sleep(_: float) -> None:
        return None

    retrying = RetryingLLMProvider(
        FlakyOnce(),
        policy=RetryPolicy(max_attempts=3, initial_delay_seconds=0.5, jitter=False),
        sleep=no_sleep,
        on_retry=RetryTelemetry(metrics, events, provider="fake", model="fake-model"),
    )
    observed = ObservedLLMProvider(retrying, metrics, events, default_model="fake-model")

    await observed.generate(REQUEST)

    assert counts(metrics)["llm_retries_total"] == 1
    assert counts(metrics)["llm_requests_total"] == 1  # retries are not logical requests
    retry_event = next(e for e in events.snapshot() if e.name == "llm.retry")
    assert retry_event.attributes["attempt"] == 1
    assert retry_event.attributes["next_attempt"] == 2
    assert retry_event.attributes["delay_seconds"] == 0.5
    assert retry_event.attributes["error_type"] == "TimeoutError"


@pytest.mark.asyncio
async def test_stream_retry_before_first_chunk_emits_retry_telemetry() -> None:
    metrics = InMemoryMetricsSink()
    events = InMemoryEventSink()

    class FlakyStream(ScriptedProvider):
        def stream(self, request: LLMRequest) -> AsyncIterator[str]:
            async def iterate() -> AsyncIterator[str]:
                self.stream_calls += 1
                if self.stream_calls == 1:
                    raise TimeoutError("transient")
                yield "ok"

            return iterate()

    async def no_sleep(_: float) -> None:
        return None

    retrying = RetryingLLMProvider(
        FlakyStream(),
        policy=RetryPolicy(max_attempts=2, jitter=False),
        sleep=no_sleep,
        on_retry=RetryTelemetry(metrics, events, provider="fake", model="fake-model"),
    )
    observed = ObservedLLMProvider(retrying, metrics, events, default_model="fake-model")

    chunks = [chunk async for chunk in observed.stream(REQUEST)]

    assert chunks == ["ok"]
    assert counts(metrics)["llm_retries_total"] == 1
    assert counts(metrics)["llm_requests_total"] == 1
    assert "llm.retry" in event_names(events)


# 5 -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_time_to_first_token_recorded_once_at_first_non_empty_chunk() -> None:
    observed, metrics, _ = observe(ScriptedProvider(chunks=("", "a", "b"), stream_delay=0.01))

    chunks = [chunk async for chunk in observed.stream(REQUEST)]

    assert chunks == ["", "a", "b"]
    ttft = [p for p in metrics.snapshot() if p.name == "llm_time_to_first_token_seconds"]
    assert len(ttft) == 1
    assert ttft[0].value >= 0.02  # measured at "a", not at the empty chunk


# 6 -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_successful_stream_completion() -> None:
    observed, metrics, events = observe(ScriptedProvider())

    chunks = [chunk async for chunk in observed.stream(REQUEST)]

    assert chunks == ["hello", " world"]
    assert counts(metrics) == Counter(
        {
            "llm_time_to_first_token_seconds": 1,
            "llm_requests_total": 1,
            "llm_request_duration_seconds": 1,
        }
    )
    assert event_names(events) == ["llm.stream.completed"]


# 7 -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_interruption_after_output() -> None:
    observed, metrics, events = observe(
        ScriptedProvider(stream_error_after=1, stream_error=ConnectionError("raw details"))
    )
    received: list[str] = []

    with pytest.raises(ConnectionError):
        async for chunk in observed.stream(REQUEST):
            received.append(chunk)

    assert received == ["hello"]
    c = counts(metrics)
    assert c["llm_stream_interruptions_total"] == 1
    assert c["llm_requests_failed_total"] == 1
    assert c["llm_requests_total"] == 1
    assert event_names(events) == ["llm.stream.interrupted"]
    assert "raw details" not in repr(events.snapshot())


@pytest.mark.asyncio
async def test_stream_failure_before_output_is_not_an_interruption() -> None:
    observed, metrics, events = observe(
        ScriptedProvider(stream_error_after=0, stream_error=ConnectionError("down"))
    )

    with pytest.raises(ConnectionError):
        async for _ in observed.stream(REQUEST):
            pass

    c = counts(metrics)
    assert c["llm_requests_failed_total"] == 1
    assert c["llm_stream_interruptions_total"] == 0
    assert event_names(events) == ["llm.request.failed"]


# 8 -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_cancellation_propagates_and_is_not_a_failure() -> None:
    observed, metrics, events = observe(ScriptedProvider(stream_delay=10))

    task = asyncio.create_task(observed.generate(REQUEST))
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    c = counts(metrics)
    assert c["llm_requests_total"] == 1
    assert c["llm_requests_cancelled_total"] == 1
    assert c["llm_requests_failed_total"] == 0
    [event] = events.snapshot()
    assert event.name == "llm.request.cancelled"
    assert event.attributes["reason"] == "cancelled"


@pytest.mark.asyncio
async def test_stream_cancellation_propagates_and_is_not_a_failure() -> None:
    observed, metrics, events = observe(ScriptedProvider(stream_delay=10))

    async def consume() -> None:
        async for _ in observed.stream(REQUEST):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    c = counts(metrics)
    assert c["llm_requests_total"] == 1
    assert c["llm_requests_cancelled_total"] == 1
    assert c["llm_requests_failed_total"] == 0
    assert c["llm_stream_interruptions_total"] == 0
    assert event_names(events) == ["llm.request.cancelled"]


# 9 -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_early_stream_closure_counted_once_as_cancelled() -> None:
    observed, metrics, events = observe(ScriptedProvider(chunks=("a", "b", "c")))

    stream = observed.stream(REQUEST)
    first = await anext(stream)
    await stream.aclose()  # type: ignore[attr-defined]

    assert first == "a"
    c = counts(metrics)
    assert c["llm_requests_total"] == 1
    assert c["llm_requests_cancelled_total"] == 1
    assert c["llm_requests_failed_total"] == 0
    [event] = events.snapshot()
    assert event.name == "llm.request.cancelled"
    assert event.attributes["reason"] == "closed_early"


# 10 ------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_empty_stream_completes_without_ttft() -> None:
    observed, metrics, events = observe(ScriptedProvider(chunks=()))

    chunks = [chunk async for chunk in observed.stream(REQUEST)]

    assert chunks == []
    c = counts(metrics)
    assert c["llm_time_to_first_token_seconds"] == 0
    assert c["llm_requests_total"] == 1
    assert event_names(events) == ["llm.stream.completed"]


# 11 ------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_missing_token_usage_records_no_token_metrics() -> None:
    response = LLMResponse(content="hi", model="fake-model", provider="fake", usage=None)
    observed, metrics, _ = observe(ScriptedProvider(response=response))

    await observed.generate(REQUEST)

    c = counts(metrics)
    assert c["llm_input_tokens_total"] == 0
    assert c["llm_output_tokens_total"] == 0
    assert c["llm_requests_total"] == 1


@pytest.mark.asyncio
async def test_partial_token_usage_records_only_reported_side() -> None:
    response = LLMResponse(
        content="hi",
        model="fake-model",
        provider="fake",
        usage=LLMUsage(input_tokens=5, output_tokens=None),
    )
    observed, metrics, _ = observe(ScriptedProvider(response=response))

    await observed.generate(REQUEST)

    c = counts(metrics)
    assert c["llm_input_tokens_total"] == 1
    assert c["llm_output_tokens_total"] == 0


# 12 ------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_metrics_sink_failure_does_not_fail_request() -> None:
    events = InMemoryEventSink()
    observed = ObservedLLMProvider(
        ScriptedProvider(),  # type: ignore[arg-type]
        BrokenMetricsSink(),
        events,
    )

    response = await observed.generate(REQUEST)
    chunks = [chunk async for chunk in observed.stream(REQUEST)]

    assert response.content == "hello"
    assert chunks == ["hello", " world"]
    assert event_names(events) == ["llm.request.completed", "llm.stream.completed"]


def test_retry_telemetry_sink_failures_are_isolated() -> None:
    telemetry = RetryTelemetry(BrokenMetricsSink(), BrokenEventSink(), "fake", "m")

    telemetry(1, 0.1, TimeoutError())  # must not raise


# 13 ------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_event_sink_failure_does_not_fail_request() -> None:
    metrics = InMemoryMetricsSink()
    observed = ObservedLLMProvider(
        ScriptedProvider(),  # type: ignore[arg-type]
        metrics,
        BrokenEventSink(),
    )

    response = await observed.generate(REQUEST)
    chunks = [chunk async for chunk in observed.stream(REQUEST)]

    assert response.content == "hello"
    assert chunks == ["hello", " world"]
    assert counts(metrics)["llm_requests_total"] == 2


# 14 ------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_correlation_id_propagates_into_events() -> None:
    observed, _, events = observe(ScriptedProvider())

    token = bind_correlation_id("req-123")
    try:
        await observed.generate(REQUEST)
        [_ async for _ in observed.stream(REQUEST)]
    finally:
        reset_correlation_id(token)

    assert [e.attributes["correlation_id"] for e in events.snapshot()] == ["req-123", "req-123"]


@pytest.mark.asyncio
async def test_concurrent_requests_keep_their_own_correlation_ids() -> None:
    observed, _, events = observe(ScriptedProvider(stream_delay=0.01))

    async def call(correlation_id: str) -> None:
        token = bind_correlation_id(correlation_id)
        try:
            await observed.generate(REQUEST)
        finally:
            reset_correlation_id(token)

    await asyncio.gather(call("a"), call("b"), call("c"))

    assert sorted(e.attributes["correlation_id"] for e in events.snapshot()) == ["a", "b", "c"]


# 15 ------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_duplicate_logical_request_counts() -> None:
    metrics = InMemoryMetricsSink()
    events = InMemoryEventSink()
    healthy = ObservedLLMProvider(ScriptedProvider(), metrics, events)  # type: ignore[arg-type]
    failing = ObservedLLMProvider(
        ScriptedProvider(stream_error_after=1, stream_error=ConnectionError()),  # type: ignore[arg-type]
        metrics,
        events,
    )

    await healthy.generate(REQUEST)  # success
    [_ async for _ in healthy.stream(REQUEST)]  # completed stream
    with pytest.raises(ConnectionError):  # interrupted stream
        async for _ in failing.stream(REQUEST):
            pass

    c = counts(metrics)
    assert c["llm_requests_total"] == 3
    assert c["llm_request_duration_seconds"] == 3
    assert c["llm_requests_failed_total"] == 1


# Model label -----------------------------------------------------------------
@pytest.mark.asyncio
async def test_model_label_falls_back_to_configured_model_never_none() -> None:
    observed, metrics, events = observe(
        ScriptedProvider(generate_error=TimeoutError()), default_model="gpt-4o-mini"
    )

    with pytest.raises(TimeoutError):
        await observed.generate(LLMRequest(messages=[], model=None))

    assert {p.labels["model"] for p in metrics.snapshot()} == {"gpt-4o-mini"}
    assert events.snapshot()[0].attributes["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_model_label_unknown_when_nothing_configured() -> None:
    observed, metrics, _ = observe(ScriptedProvider(chunks=("x",)), default_model=None)

    [_ async for _ in observed.stream(LLMRequest(messages=[]))]

    assert {p.labels["model"] for p in metrics.snapshot()} == {"unknown"}


@pytest.mark.asyncio
async def test_request_model_takes_precedence_over_default() -> None:
    observed, metrics, _ = observe(ScriptedProvider(chunks=("x",)), default_model="default")

    [_ async for _ in observed.stream(LLMRequest(messages=[], model="override"))]

    assert {p.labels["model"] for p in metrics.snapshot()} == {"override"}


# Bounded sinks ---------------------------------------------------------------
def test_in_memory_sinks_are_bounded() -> None:
    metrics = InMemoryMetricsSink(max_points=3)
    events = InMemoryEventSink(max_events=2)

    for index in range(5):
        metrics.record(MetricPoint(name=f"m{index}", value=1, labels={}))
        events.emit(ObservabilityEvent(name=f"e{index}", attributes={}))

    assert [p.name for p in metrics.snapshot()] == ["m2", "m3", "m4"]
    assert [e.name for e in events.snapshot()] == ["e3", "e4"]


@pytest.mark.parametrize("size", [0, -1])
def test_in_memory_sinks_reject_invalid_bounds(size: int) -> None:
    with pytest.raises(ValueError):
        InMemoryMetricsSink(max_points=size)
    with pytest.raises(ValueError):
        InMemoryEventSink(max_events=size)
