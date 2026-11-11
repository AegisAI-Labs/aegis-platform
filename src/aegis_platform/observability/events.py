from threading import Lock

from .contracts import ObservabilityEvent


class InMemoryEventSink:
    """Thread-safe event sink for tests and local development."""

    def __init__(self) -> None:
        self._events: list[ObservabilityEvent] = []
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
