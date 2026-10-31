"""Run deterministic local experiments for provider retry behavior."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections.abc import AsyncIterator

from aegis_platform.llm.errors import ProviderTimeout, ProviderUnavailable
from aegis_platform.llm.models import LLMMessage, LLMRequest, LLMResponse
from aegis_platform.llm.reliability import RetryingLLMProvider, RetryPolicy

PROMPT = "Summarize retry behavior in one sentence."


class ScriptedProvider:
    """Provider that produces repeatable failures without network access."""

    provider_name = "scripted"

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.generate_calls = 0
        self.stream_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        if self.scenario == "generate_retry" and self.generate_calls == 1:
            raise ProviderTimeout("scripted timeout before response")
        if self.scenario == "generate_exhaustion":
            raise ProviderUnavailable("scripted provider unavailable")
        return LLMResponse(
            content="recovered response",
            model="scripted-model",
            provider=self.provider_name,
        )

    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        async def chunks() -> AsyncIterator[str]:
            self.stream_calls += 1
            if self.scenario == "stream_retry" and self.stream_calls == 1:
                raise ProviderTimeout("scripted timeout before first chunk")
            if self.scenario == "stream_partial":
                yield "partial output"
                raise ProviderTimeout("scripted timeout after first chunk")
            yield "recovered stream"

        return chunks()


def make_retrying_provider(provider: ScriptedProvider) -> RetryingLLMProvider:
    return RetryingLLMProvider(
        provider,
        policy=RetryPolicy(
            max_attempts=3,
            timeout_seconds=5,
            initial_delay_seconds=0.001,
            max_delay_seconds=0.002,
            jitter=False,
        ),
        retryable_exceptions=(ProviderTimeout, ProviderUnavailable),
    )


def create_request() -> LLMRequest:
    return LLMRequest(messages=[LLMMessage(role="user", content=PROMPT)])


def report_timing(name: str, durations: list[float]) -> None:
    print(
        f"  {name}: iterations={len(durations)}, "
        f"mean={statistics.mean(durations):.6f}s, "
        f"max={max(durations):.6f}s"
    )


async def run_experiments(iterations: int) -> bool:
    print("Retry reliability experiment (scripted provider; no network)")
    print(f"Iterations per scenario: {iterations}")
    passed = True

    generate_durations: list[float] = []
    generate_attempts = 0
    for _ in range(iterations):
        provider = ScriptedProvider("generate_retry")
        retrying_provider = make_retrying_provider(provider)
        started = time.perf_counter()
        response = await retrying_provider.generate(create_request())
        generate_durations.append(time.perf_counter() - started)
        generate_attempts += provider.generate_calls
        passed = passed and response.content == "recovered response"
        passed = passed and provider.generate_calls == 2
    print("\nGenerate retries a transient timeout:")
    report_timing("recovery", generate_durations)
    print(f"  attempts={generate_attempts}; expected={iterations * 2}")
    print(f"  result={'PASS' if passed else 'FAIL'}")

    exhaustion_attempts = 0
    exhaustion_passed = True
    for _ in range(iterations):
        provider = ScriptedProvider("generate_exhaustion")
        retrying_provider = make_retrying_provider(provider)
        try:
            await retrying_provider.generate(create_request())
        except ProviderUnavailable:
            exhaustion_passed = exhaustion_passed and provider.generate_calls == 3
        else:
            exhaustion_passed = False
        exhaustion_attempts += provider.generate_calls
    passed = passed and exhaustion_passed
    print("\nGenerate stops at max_attempts when failures persist:")
    print(f"  attempts={exhaustion_attempts}; expected={iterations * 3}")
    print(f"  result={'PASS' if exhaustion_passed else 'FAIL'}")

    stream_durations: list[float] = []
    stream_attempts = 0
    stream_passed = True
    for _ in range(iterations):
        provider = ScriptedProvider("stream_retry")
        retrying_provider = make_retrying_provider(provider)
        started = time.perf_counter()
        chunks = [chunk async for chunk in retrying_provider.stream(create_request())]
        stream_durations.append(time.perf_counter() - started)
        stream_attempts += provider.stream_calls
        stream_passed = stream_passed and chunks == ["recovered stream"]
        stream_passed = stream_passed and provider.stream_calls == 2
    passed = passed and stream_passed
    print("\nStream retries a timeout before the first chunk:")
    report_timing("recovery", stream_durations)
    print(f"  attempts={stream_attempts}; expected={iterations * 2}")
    print(f"  result={'PASS' if stream_passed else 'FAIL'}")

    partial_attempts = 0
    partial_passed = True
    for _ in range(iterations):
        provider = ScriptedProvider("stream_partial")
        retrying_provider = make_retrying_provider(provider)
        chunks: list[str] = []
        try:
            async for chunk in retrying_provider.stream(create_request()):
                chunks.append(chunk)
        except ProviderTimeout:
            partial_passed = partial_passed and chunks == ["partial output"]
            partial_passed = partial_passed and provider.stream_calls == 1
        else:
            partial_passed = False
        partial_attempts += provider.stream_calls
    passed = passed and partial_passed
    print("\nStream does not retry after delivering partial output:")
    print(f"  attempts={partial_attempts}; expected={iterations}")
    print(f"  result={'PASS' if partial_passed else 'FAIL'}")

    print(f"\nOverall: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if not asyncio.run(run_experiments(args.iterations)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
