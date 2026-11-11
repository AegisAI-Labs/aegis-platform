from threading import Lock

from .contracts import MetricPoint


class InMemoryMetricsSink:
    """Thread-safe metrics sink for tests and local development."""

    def __init__(self) -> None:
        self._points: list[MetricPoint] = []
        self._lock = Lock()

    def record(self, point: MetricPoint) -> None:
        with self._lock:
            self._points.append(point)

    def snapshot(self) -> list[MetricPoint]:
        with self._lock:
            return list(self._points)

    def clear(self) -> None:
        with self._lock:
            self._points.clear()
