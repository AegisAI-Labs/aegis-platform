from collections import deque
from threading import Lock

from .contracts import ObservabilityEvent

DEFAULT_MAX_EVENTS = 10_000


class InMemoryEventSink:
    """Thread-safe, bounded event sink for tests and local development.

    Keeps the most recent ``max_events`` events and drops the oldest.
    """

    def __init__(self, max_events: int = DEFAULT_MAX_EVENTS) -> None:
        if max_events < 1:
            raise ValueError("max_events must be >= 1")
        self._events: deque[ObservabilityEvent] = deque(maxlen=max_events)
        self._lock = Lock()

    def emit(self, event: ObservabilityEvent) -> None:
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> list[ObservabilityEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
