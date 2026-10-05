"""
Aegis AI Platform
Day 4 — LLM Provider Latency Experiment

Measures:

1. Calculator baseline
2. Fake LLM provider
3. Real LLM provider - generate()
4. Real LLM provider - stream()

The experiment is intentionally lightweight. It establishes a Day 4
performance baseline and is not intended to replace the comprehensive
load/performance testing planned for later phases of Aegis.
"""

from __future__ import annotations

import asyncio
import math
import statistics
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from aegis_platform.agent.service import AgentService
from aegis_platform.config.settings import settings
from aegis_platform.llm.base import LLMProvider
from aegis_platform.llm.factory import LLMProviderFactory
from aegis_platform.llm.models import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMUsage,
)

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

ITERATIONS = 10

TEST_PROMPT = "Explain what an event-driven architecture is in three concise sentences."

CALCULATOR_EXPRESSION = "25 * 17"


# ---------------------------------------------------------------------------
# Measurement models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyStats:
    """Statistical summary for a collection of latency measurements."""

    iterations: int
    minimum: float
    median: float
    mean: float
    p95: float
    maximum: float


@dataclass(frozen=True)
class StreamMeasurement:
    """Measurement for a single streaming execution."""

    time_to_first_chunk: float
    total_latency: float


@dataclass(frozen=True)
class StreamStats:
    """Statistical summary for streaming measurements."""

    iterations: int
    ttfc_minimum: float
    ttfc_median: float
    ttfc_mean: float
    ttfc_p95: float
    ttfc_maximum: float
    total_minimum: float
    total_median: float
    total_mean: float
    total_p95: float
    total_maximum: float


# ---------------------------------------------------------------------------
# Fake provider
# ---------------------------------------------------------------------------


class FakeLLMProvider:
    """
    Deterministic provider used to measure Aegis-side provider abstraction
    overhead without network or external model latency.

    This class intentionally implements only the behavior required by this
    experiment.
    """

    provider_name = "fake"

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        return LLMResponse(
            content="This is a deterministic fake LLM response.",
            model="fake-model",
            provider=self.provider_name,
            usage=LLMUsage(
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
            ),
        )

    def stream(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[str]:
        async def _generate() -> AsyncIterator[str]:
            chunks = (
                "This ",
                "is ",
                "a ",
                "deterministic ",
                "fake ",
                "LLM ",
                "response.",
            )

            for chunk in chunks:
                yield chunk

        return _generate()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def percentile(values: list[float], percentile_value: float) -> float:
    """
    Calculate a percentile using linear interpolation.

    Example:
        percentile(values, 95)
    """

    if not values:
        raise ValueError("Cannot calculate percentile from empty values.")

    if len(values) == 1:
        return values[0]

    ordered = sorted(values)

    position = (len(ordered) - 1) * (percentile_value / 100.0)

    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return ordered[lower_index]

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]

    weight = position - lower_index

    return lower_value + (upper_value - lower_value) * weight


def calculate_latency_stats(
    values: list[float],
) -> LatencyStats:
    """Calculate standard latency statistics."""

    if not values:
        raise ValueError("No latency measurements were collected.")

    return LatencyStats(
        iterations=len(values),
        minimum=min(values),
        median=statistics.median(values),
        mean=statistics.mean(values),
        p95=percentile(values, 95),
        maximum=max(values),
    )


def calculate_stream_stats(
    measurements: list[StreamMeasurement],
) -> StreamStats:
    """Calculate streaming latency statistics."""

    if not measurements:
        raise ValueError("No streaming measurements were collected.")

    ttfc_values = [measurement.time_to_first_chunk for measurement in measurements]

    total_values = [measurement.total_latency for measurement in measurements]

    return StreamStats(
        iterations=len(measurements),
        ttfc_minimum=min(ttfc_values),
        ttfc_median=statistics.median(ttfc_values),
        ttfc_mean=statistics.mean(ttfc_values),
        ttfc_p95=percentile(ttfc_values, 95),
        ttfc_maximum=max(ttfc_values),
        total_minimum=min(total_values),
        total_median=statistics.median(total_values),
        total_mean=statistics.mean(total_values),
        total_p95=percentile(total_values, 95),
        total_maximum=max(total_values),
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_seconds(value: float) -> str:
    """Format seconds consistently for human-readable output."""

    return f"{value:.4f} s"


def print_separator() -> None:
    print("=" * 72)


def print_latency_stats(
    name: str,
    stats: LatencyStats,
) -> None:
    """Print latency statistics."""

    print_separator()
    print(name)
    print_separator()

    print(f"Iterations : {stats.iterations}")
    print(f"Min        : {format_seconds(stats.minimum)}")
    print(f"Median     : {format_seconds(stats.median)}")
    print(f"Mean       : {format_seconds(stats.mean)}")
    print(f"P95        : {format_seconds(stats.p95)}")
    print(f"Max        : {format_seconds(stats.maximum)}")


def print_stream_stats(
    stats: StreamStats,
) -> None:
    """Print streaming statistics."""

    print_separator()
    print("Real LLM — Stream")
    print_separator()

    print(f"Iterations       : {stats.iterations}")

    print()
    print("Time to First Chunk")
    print(f"Min              : {format_seconds(stats.ttfc_minimum)}")
    print(f"Median           : {format_seconds(stats.ttfc_median)}")
    print(f"Mean             : {format_seconds(stats.ttfc_mean)}")
    print(f"P95              : {format_seconds(stats.ttfc_p95)}")
    print(f"Max              : {format_seconds(stats.ttfc_maximum)}")

    print()
    print("Total Streaming Latency")
    print(f"Min              : {format_seconds(stats.total_minimum)}")
    print(f"Median           : {format_seconds(stats.total_median)}")
    print(f"Mean             : {format_seconds(stats.total_mean)}")
    print(f"P95              : {format_seconds(stats.total_p95)}")
    print(f"Max              : {format_seconds(stats.total_maximum)}")


# ---------------------------------------------------------------------------
# Scenario A — Calculator
# ---------------------------------------------------------------------------


async def run_calculator_iteration(service: AgentService) -> float:
    """Measure one calculator execution."""

    start = time.perf_counter()

    result = await service.run(f"calculate {CALCULATOR_EXPRESSION}")

    end = time.perf_counter()

    if result != {"response": "425", "tool_used": "calculator"}:
        raise RuntimeError(f"Unexpected calculator result: {result}")

    return end - start


async def run_calculator_experiment() -> LatencyStats:
    """Run the calculator latency experiment."""

    service = AgentService(FakeLLMProvider())
    measurements: list[float] = []

    print()
    print("Running Calculator baseline...")

    for iteration in range(1, ITERATIONS + 1):
        latency = await run_calculator_iteration(service)

        measurements.append(latency)

        print(f"Run {iteration:02d}: " f"{format_seconds(latency)}")

    return calculate_latency_stats(measurements)


# ---------------------------------------------------------------------------
# Scenario B — Fake LLM Provider
# ---------------------------------------------------------------------------


def create_llm_request() -> LLMRequest:
    """Create the deterministic request used by the experiment."""

    return LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content=TEST_PROMPT,
            )
        ],
    )


async def run_fake_generate_iteration(
    service: AgentService,
) -> float:
    """Measure one fake-provider request through AgentService."""

    start = time.perf_counter()

    result = await service.run(TEST_PROMPT)

    end = time.perf_counter()

    if result != {"response": "This is a deterministic fake LLM response.", "tool_used": "llm"}:
        raise RuntimeError(f"Unexpected fake LLM result: {result}")

    return end - start


async def run_fake_llm_experiment() -> LatencyStats:
    """Run the FakeLLMProvider latency experiment."""

    service = AgentService(FakeLLMProvider())

    measurements: list[float] = []

    print()
    print("Running Fake LLM provider...")

    for iteration in range(1, ITERATIONS + 1):
        latency = await run_fake_generate_iteration(service)

        measurements.append(latency)

        print(f"Run {iteration:02d}: " f"{format_seconds(latency)}")

    return calculate_latency_stats(measurements)


# ---------------------------------------------------------------------------
# Scenario C — Real LLM generate()
# ---------------------------------------------------------------------------


async def run_real_generate_iteration(
    service: AgentService,
) -> float:
    """Measure one real-provider request through AgentService."""

    start = time.perf_counter()

    result = await service.run(TEST_PROMPT)

    end = time.perf_counter()

    if result["tool_used"] != "llm":
        raise RuntimeError(f"Unexpected real LLM route: {result}")

    return end - start


async def run_real_generate_experiment(
    provider: LLMProvider,
) -> LatencyStats:
    """Run the real-provider generate() experiment."""

    service = AgentService(provider)
    measurements: list[float] = []

    print()
    print("Running real LLM generate()...")

    for iteration in range(1, ITERATIONS + 1):
        latency = await run_real_generate_iteration(service)

        measurements.append(latency)

        print(f"Run {iteration:02d}: " f"{format_seconds(latency)}")

    return calculate_latency_stats(measurements)


# ---------------------------------------------------------------------------
# Scenario D — Real LLM stream()
# ---------------------------------------------------------------------------


async def run_stream_iteration(
    provider: LLMProvider,
) -> StreamMeasurement:
    """
    Measure one streaming execution.

    Metrics:

    TTFC:
        Time from request start until the first non-empty chunk.

    Total latency:
        Time from request start until the stream is exhausted.
    """

    request = create_llm_request()

    start = time.perf_counter()

    first_chunk_time: float | None = None

    async for chunk in provider.stream(request):
        now = time.perf_counter()

        if first_chunk_time is None and chunk:
            first_chunk_time = now

    end = time.perf_counter()

    if first_chunk_time is None:
        raise RuntimeError("Streaming provider returned no non-empty chunks.")

    return StreamMeasurement(
        time_to_first_chunk=first_chunk_time - start,
        total_latency=end - start,
    )


async def run_real_stream_experiment(
    provider: LLMProvider,
) -> StreamStats:
    """Run the real-provider streaming experiment."""

    measurements: list[StreamMeasurement] = []

    print()
    print("Running real LLM stream()...")

    for iteration in range(1, ITERATIONS + 1):
        measurement = await run_stream_iteration(provider)

        measurements.append(measurement)

        print(
            f"Run {iteration:02d}: "
            f"TTFC={format_seconds(measurement.time_to_first_chunk)}, "
            f"Total={format_seconds(measurement.total_latency)}"
        )

    return calculate_stream_stats(measurements)


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def markdown_latency_table(
    stats: LatencyStats,
) -> str:
    """Create a Markdown result table."""

    return "\n".join(
        [
            "| Metric | Result |",
            "|---|---:|",
            f"| Iterations | {stats.iterations} |",
            f"| Min | {format_seconds(stats.minimum)} |",
            f"| Median | {format_seconds(stats.median)} |",
            f"| Mean | {format_seconds(stats.mean)} |",
            f"| P95 | {format_seconds(stats.p95)} |",
            f"| Max | {format_seconds(stats.maximum)} |",
        ]
    )


def markdown_stream_table(
    stats: StreamStats,
) -> str:
    """Create a Markdown streaming result table."""

    return "\n".join(
        [
            "| Metric | Result |",
            "|---|---:|",
            f"| Iterations | {stats.iterations} |",
            f"| Min TTFC | {format_seconds(stats.ttfc_minimum)} |",
            f"| Median TTFC | {format_seconds(stats.ttfc_median)} |",
            f"| Mean TTFC | {format_seconds(stats.ttfc_mean)} |",
            f"| P95 TTFC | {format_seconds(stats.ttfc_p95)} |",
            f"| Max TTFC | {format_seconds(stats.ttfc_maximum)} |",
            f"| Min Total | {format_seconds(stats.total_minimum)} |",
            f"| Median Total | {format_seconds(stats.total_median)} |",
            f"| Mean Total | {format_seconds(stats.total_mean)} |",
            f"| P95 Total | {format_seconds(stats.total_p95)} |",
            f"| Max Total | {format_seconds(stats.total_maximum)} |",
        ]
    )


def print_markdown_summary(
    calculator_stats: LatencyStats,
    fake_stats: LatencyStats,
    real_generate_stats: LatencyStats,
    stream_stats: StreamStats,
) -> None:
    """Print a Markdown-ready summary."""

    print()
    print()
    print("# Markdown Summary")
    print()

    print("## Calculator")
    print()
    print(markdown_latency_table(calculator_stats))
    print()

    print("## Fake LLM")
    print()
    print(markdown_latency_table(fake_stats))
    print()

    print("## Real LLM — Generate")
    print()
    print(markdown_latency_table(real_generate_stats))
    print()

    print("## Real LLM — Stream")
    print()
    print(markdown_stream_table(stream_stats))
    print()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


async def main() -> None:
    """Execute all Day 4 latency experiments."""

    if not settings.llm_model:
        raise ValueError(
            "Set LLM_MODEL to a valid model before running the real-provider experiment."
        )

    provider = LLMProviderFactory.create_llm_provider(settings)

    print_separator()
    print("Aegis AI Platform")
    print("Day 4 — LLM Provider Latency Experiment")
    print_separator()

    print()
    print(f"Iterations : {ITERATIONS}")
    print(f"Prompt     : {TEST_PROMPT}")
    print(f"Provider   : " f"{settings.llm_provider}")
    print(f"Model      : " f"{settings.llm_model or '<provider default>'}")

    # ---------------------------------------------------------------
    # Scenario A
    # ---------------------------------------------------------------

    calculator_stats = await run_calculator_experiment()

    print_latency_stats(
        "Calculator",
        calculator_stats,
    )

    # ---------------------------------------------------------------
    # Scenario B
    # ---------------------------------------------------------------

    fake_stats = await run_fake_llm_experiment()

    print_latency_stats(
        "Fake LLM",
        fake_stats,
    )

    # ---------------------------------------------------------------
    # Create real provider
    # ---------------------------------------------------------------

    print()
    print("Creating configured real LLM provider...")

    print(f"Provider implementation: " f"{type(provider).__name__}")

    # ---------------------------------------------------------------
    # Scenario C
    # ---------------------------------------------------------------

    real_generate_stats = await run_real_generate_experiment(
        provider,
    )

    print_latency_stats(
        "Real LLM — Generate",
        real_generate_stats,
    )

    # ---------------------------------------------------------------
    # Scenario D
    # ---------------------------------------------------------------

    stream_stats = await run_real_stream_experiment(
        provider,
    )

    print_stream_stats(stream_stats)

    # ---------------------------------------------------------------
    # Markdown summary
    # ---------------------------------------------------------------

    print_markdown_summary(
        calculator_stats=calculator_stats,
        fake_stats=fake_stats,
        real_generate_stats=real_generate_stats,
        stream_stats=stream_stats,
    )

    print_separator()
    print("Experiment complete.")
    print_separator()


if __name__ == "__main__":
    asyncio.run(main())
