from collections import deque
from threading import Lock

from .contracts import MetricPoint

DEFAULT_MAX_POINTS = 10_000


class InMemoryMetricsSink:
    """Thread-safe, bounded metrics sink for tests and local development.

    Keeps the most recent ``max_points`` points and drops the oldest, so a long-running
    process cannot grow memory without limit.
    """

    def __init__(self, max_points: int = DEFAULT_MAX_POINTS) -> None:
        if max_points < 1:
            raise ValueError("max_points must be >= 1")
        self._points: deque[MetricPoint] = deque(maxlen=max_points)
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
