"""Measure OpenAI provider streaming behavior and reliability scenarios."""

from __future__ import annotations

import argparse
import asyncio
import math
import platform
import statistics
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

from httpx2 import Request, Response
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)

from aegis_platform.agent.service import AgentService
from aegis_platform.config.settings import settings
from aegis_platform.llm.errors import (
    LLMProviderError,
    ProviderAuthenticationError,
    ProviderRateLimited,
    ProviderRequestError,
    ProviderTimeout,
    ProviderUnavailable,
)
from aegis_platform.llm.models import LLMMessage, LLMRequest, LLMResponse, LLMUsage
from aegis_platform.llm.providers.openai import OpenAIProvider

DEFAULT_ITERATIONS = 10
PROMPT = "Explain what an event-driven architecture is in three concise sentences."
CALCULATOR_EXPRESSION = "25 * 17"


@dataclass(frozen=True)
class StreamMeasurement:
    time_to_first_chunk: float
    total_duration: float
    chunks_received: int


class FakeLLMProvider:
    provider_name = "fake"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            content="This is a deterministic fake LLM response.",
            model="fake-model",
            provider=self.provider_name,
            usage=LLMUsage(),
        )

    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        async def generate_chunks() -> AsyncIterator[str]:
            for chunk in ("This ", "is ", "a ", "deterministic ", "fake ", "LLM ", "response."):
                yield chunk

        return generate_chunks()


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def print_latency_summary(name: str, values: list[float], failures: int = 0) -> None:
    if not values:
        print(f"  {name}: no successful measurements; failures={failures}")
        return
    print(
        f"  {name}: successful={len(values)}, failures={failures}, "
        f"min={format_seconds(min(values))}, "
        f"median={format_seconds(statistics.median(values))}, "
        f"mean={format_seconds(statistics.mean(values))}, "
        f"p95={format_seconds(percentile(values, 95))}, "
        f"max={format_seconds(max(values))}"
    )


async def run_local_scenarios(iterations: int) -> None:
    print(f"\nCalculator baseline ({iterations} iterations):")
    calculator = AgentService(FakeLLMProvider())
    calculator_times: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        result = await calculator.run(f"calculate {CALCULATOR_EXPRESSION}")
        calculator_times.append(time.perf_counter() - started)
        if result != {"response": "425", "tool_used": "calculator"}:
            raise RuntimeError(f"Unexpected calculator result: {result}")
    print_latency_summary("Calculator", calculator_times)

    print(f"\nFake LLM through AgentService/LangGraph ({iterations} iterations):")
    fake_service = AgentService(FakeLLMProvider())
    fake_times: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        result = await fake_service.run(PROMPT)
        fake_times.append(time.perf_counter() - started)
        if result != {
            "response": "This is a deterministic fake LLM response.",
            "tool_used": "llm",
        }:
            raise RuntimeError(f"Unexpected fake provider result: {result}")
    print_latency_summary("Fake LLM", fake_times)


def create_request() -> LLMRequest:
    return LLMRequest(
        messages=[LLMMessage(role="user", content=PROMPT)],
        model=settings.llm_model,
    )


def sdk_version() -> str:
    try:
        return version("openai")
    except PackageNotFoundError:
        return "not installed"


def error_mapping_cases() -> list[tuple[str, OpenAIError, type[Exception]]]:
    request = Request("POST", "https://example.test/chat/completions")
    return [
        (
            "authentication",
            AuthenticationError("unauthorized", response=Response(401, request=request), body=None),
            ProviderAuthenticationError,
        ),
        (
            "rate limit",
            RateLimitError("rate limited", response=Response(429, request=request), body=None),
            ProviderRateLimited,
        ),
        ("timeout", APITimeoutError(request=request), ProviderTimeout),
        ("connection", APIConnectionError(request=request), ProviderUnavailable),
        ("unclassified SDK error", OpenAIError("unexpected SDK error"), ProviderRequestError),
    ]


def run_error_mapping_scenarios() -> bool:
    print("\nError mapping scenarios (synthetic SDK exceptions; no network):")
    passed = True
    for name, sdk_error, expected_type in error_mapping_cases():
        mapped_error = OpenAIProvider._map_error(sdk_error)
        matches = isinstance(mapped_error, expected_type)
        passed = passed and matches
        result = "PASS" if matches else "FAIL"
        print(f"  {name}: {type(sdk_error).__name__} -> {type(mapped_error).__name__} [{result}]")

    request = Request("POST", "https://example.test/chat/completions")
    quota_sdk_error = RateLimitError(
        "quota exhausted",
        response=Response(429, request=request),
        body={"code": "credit_balance_exhausted"},
    )
    quota_error = OpenAIProvider._map_error(quota_sdk_error)
    assert isinstance(quota_error, ProviderRateLimited)
    quota_error.__cause__ = quota_sdk_error
    quota_classification = describe_error(quota_error)
    quota_matches = quota_classification == "quota exhausted (credit_balance_exhausted)"
    passed = passed and quota_matches
    result = "PASS" if quota_matches else "FAIL"
    print(f"  exhausted quota: {quota_classification} [{result}]")
    return passed


def describe_error(error: LLMProviderError) -> str:
    cause = error.__cause__
    if (
        isinstance(error, ProviderRateLimited)
        and isinstance(cause, RateLimitError)
        and cause.code in {"insufficient_quota", "credit_balance_exhausted"}
    ):
        return f"quota exhausted ({cause.code})"
    return type(error).__name__


def is_quota_exhausted(error: LLMProviderError) -> bool:
    return describe_error(error).startswith("quota exhausted")


async def run_generate_scenario(provider: OpenAIProvider, iterations: int) -> bool:
    print(f"\nReal LLM generate() through AgentService/LangGraph ({iterations} iterations):")
    service = AgentService(provider)
    measurements: list[float] = []
    failures = 0
    quota_exhausted = False

    for iteration in range(1, iterations + 1):
        started = time.perf_counter()
        try:
            result = await service.run(PROMPT)
        except LLMProviderError as error:
            duration = time.perf_counter() - started
            classification = describe_error(error)
            failures += 1
            print(f"  run {iteration}: error={classification}, total={format_seconds(duration)}")
            if is_quota_exhausted(error):
                print("  stopping live requests because the account has no usable quota")
                quota_exhausted = True
                break
        else:
            duration = time.perf_counter() - started
            if result["tool_used"] != "llm":
                raise RuntimeError(f"Unexpected real LLM route: {result}")
            measurements.append(duration)
            print(f"  run {iteration}: total={format_seconds(duration)}, error=none")

    print_latency_summary("Generate latency", measurements, failures)
    return quota_exhausted


async def measure_stream(
    provider: OpenAIProvider,
    iteration: int,
) -> StreamMeasurement | LLMProviderError:
    request = create_request()
    started = time.perf_counter()
    first_chunk_at: float | None = None
    chunks_received = 0

    try:
        async for chunk in provider.stream(request):
            if chunk:
                chunks_received += 1
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
    except LLMProviderError as error:
        total_duration = time.perf_counter() - started
        print(
            f"  run {iteration}: TTFC=not measured, total={format_seconds(total_duration)}, "
            f"chunks={chunks_received}, error={describe_error(error)}"
        )
        return error

    total_duration = time.perf_counter() - started
    time_to_first_chunk = first_chunk_at - started if first_chunk_at is not None else None
    print(
        f"  run {iteration}: TTFC={format_seconds(time_to_first_chunk)}, "
        f"total={format_seconds(total_duration)}, chunks={chunks_received}, error=none"
    )
    if time_to_first_chunk is None:
        return ProviderRequestError("Streaming provider returned no non-empty chunks.")
    return StreamMeasurement(time_to_first_chunk, total_duration, chunks_received)


async def run_stream_scenario(provider: OpenAIProvider, iterations: int) -> bool:
    print(f"\nReal LLM provider.stream() ({iterations} iterations):")
    measurements: list[StreamMeasurement] = []
    failures = 0
    quota_exhausted = False
    for iteration in range(1, iterations + 1):
        measurement = await measure_stream(provider, iteration)
        if isinstance(measurement, LLMProviderError):
            failures += 1
            if is_quota_exhausted(measurement):
                print("  stopping live requests because the account has no usable quota")
                quota_exhausted = True
                break
        else:
            measurements.append(measurement)

    print_latency_summary(
        "TTFC",
        [measurement.time_to_first_chunk for measurement in measurements],
        failures,
    )
    print_latency_summary(
        "Total stream duration",
        [measurement.total_duration for measurement in measurements],
        failures,
    )
    if measurements:
        print(
            "  chunks received: "
            f"min={min(item.chunks_received for item in measurements)}, "
            f"median={statistics.median(item.chunks_received for item in measurements)}, "
            f"max={max(item.chunks_received for item in measurements)}"
        )
    return quota_exhausted


def format_seconds(value: float | None) -> str:
    return f"{value:.4f}s" if value is not None else "not measured"


async def measure_cancellation(provider: OpenAIProvider) -> None:
    print("\nCancellation scenario (cancel consumer after first non-empty chunk):")
    request = create_request()
    first_chunk = asyncio.Event()
    started = time.perf_counter()
    chunks_received = 0
    first_chunk_at: float | None = None

    async def consume() -> None:
        nonlocal chunks_received, first_chunk_at
        async for chunk in provider.stream(request):
            if chunk:
                chunks_received += 1
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                    first_chunk.set()

    stream_task = asyncio.create_task(consume())
    first_chunk_waiter = asyncio.create_task(first_chunk.wait())
    timeout = max(settings.llm_timeout_seconds + 5.0, 30.0)
    done, _ = await asyncio.wait(
        {stream_task, first_chunk_waiter},
        timeout=timeout,
        return_when=asyncio.FIRST_COMPLETED,
    )

    if first_chunk_waiter not in done:
        if stream_task.done():
            try:
                stream_task.result()
                reason = "stream ended without a non-empty chunk"
            except LLMProviderError as error:
                reason = f"stream failed before first chunk: {describe_error(error)}"
        else:
            reason = f"no first chunk within {timeout:.1f}s"
        first_chunk_waiter.cancel()
        with suppress(asyncio.CancelledError):
            await first_chunk_waiter
        if not stream_task.done():
            stream_task.cancel()
            with suppress(asyncio.CancelledError):
                await stream_task
        print(f"  cancellation not exercised: {reason}")
        print(f"  chunks received before exit: {chunks_received}")
        return

    first_chunk_waiter.result()
    first_chunk_waiter.cancel()
    with suppress(asyncio.CancelledError):
        await first_chunk_waiter

    cancellation_requested = time.perf_counter()
    stream_task.cancel()
    cancellation_propagated = False
    error_result = "none"
    try:
        await stream_task
    except asyncio.CancelledError:
        cancellation_propagated = True
    except LLMProviderError as error:
        error_result = describe_error(error)

    finished = time.perf_counter()
    ttfc = first_chunk_at - started if first_chunk_at is not None else None
    cancellation_latency = finished - cancellation_requested
    total_duration = finished - started
    print(f"  TTFC: {format_seconds(ttfc)}")
    print(f"  chunks received before cancellation: {chunks_received}")
    print(f"  cancellation propagated: {cancellation_propagated}")
    print(f"  cancellation-to-task-exit: {format_seconds(cancellation_latency)}")
    print(f"  total duration until task exit: {format_seconds(total_duration)}")
    print(f"  error classification: {error_result}")
    print("  remote generation stopped: not observable from this client")


async def run_live_scenarios(iterations: int) -> None:
    if not settings.openai_api_key.strip() or settings.openai_api_key == "your-real-key":
        print("\nLive LLM scenarios skipped: configure OPENAI_API_KEY to run them.")
        return
    if not settings.llm_model.strip():
        print("\nLive LLM scenarios skipped: configure LLM_MODEL to run them.")
        return

    provider = OpenAIProvider(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    try:
        quota_exhausted = await run_generate_scenario(provider, iterations)
        if quota_exhausted:
            print("\nProvider streaming and cancellation scenarios skipped after quota exhaustion.")
            return

        quota_exhausted = await run_stream_scenario(provider, iterations)
        if quota_exhausted:
            print("\nCancellation scenario skipped after quota exhaustion.")
            return

        await measure_cancellation(provider)
    finally:
        await provider.client.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="run real OpenAI generate, stream, and cancellation scenarios",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"iteration count for local and live scenarios (default: {DEFAULT_ITERATIONS})",
    )
    arguments = parser.parse_args()
    if arguments.iterations < 1:
        parser.error("--iterations must be at least 1")

    print("Aegis Day 5 — Streaming Reliability Experiment")
    print(f"Python version: {platform.python_version()}")
    print(f"OpenAI SDK version: {sdk_version()}")
    print(f"Model: {settings.llm_model}")
    print(f"Iterations: {arguments.iterations}")
    print(f"Prompt: {PROMPT}")
    print(f"Live API calls enabled: {arguments.live}")

    mapping_passed = run_error_mapping_scenarios()
    await run_local_scenarios(arguments.iterations)
    if arguments.live:
        await run_live_scenarios(arguments.iterations)
    else:
        print("\nLive LLM scenarios not run; pass --live to enable paid API calls.")

    if not mapping_passed:
        raise SystemExit("At least one SDK-to-Aegis error mapping check failed.")


if __name__ == "__main__":
    asyncio.run(main())
