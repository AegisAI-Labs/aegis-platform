import logging

from .context import get_correlation_id
from .contracts import (
    EventSink,
    MetricPoint,
    MetricsSink,
    ObservabilityEvent,
)

logger = logging.getLogger(__name__)


class RetryTelemetry:
    def __init__(
        self,
        metrics: MetricsSink,
        events: EventSink,
        provider: str,
        model: str,
    ) -> None:
        self._metrics = metrics
        self._events = events
        self._provider = provider
        self._model = model

    def __call__(
        self,
        attempt: int,
        delay_seconds: float,
        exc: Exception,
    ) -> None:
        try:
            self._metrics.record(
                MetricPoint(
                    name="llm_retries_total",
                    value=1,
                    labels={
                        "provider": self._provider,
                        "model": self._model,
                    },
                )
            )

            self._events.emit(
                ObservabilityEvent(
                    name="llm.retry",
                    attributes={
                        "provider": self._provider,
                        "model": self._model,
                        "attempt": attempt,
                        "delay_seconds": delay_seconds,
                        "error_type": type(exc).__name__,
                        "correlation_id": get_correlation_id(),
                    },
                )
            )

        except Exception:
            logger.warning("Unable to record retry telemetry")
