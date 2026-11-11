## These interfaces deliberately avoid Prometheus or OpenTelemetry dependencies. We can add those backends later without changing the LLM provider contract.
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class MetricPoint:
    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)


class MetricsSink(Protocol):
    def record(self, point: MetricPoint) -> None: ...


@dataclass(frozen=True)
class ObservabilityEvent:
    name: str
    attributes: dict[str, str | int | float | bool]


class EventSink(Protocol):
    def emit(self, event: ObservabilityEvent) -> None: ...
