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
    """``RetryingLLMProvider`` ``on_retry`` callback: one metric point and one event per retry.

    Best-effort: sink failures are logged and never affect retry behavior.
    """

    def __init__(
        self,
        metrics: MetricsSink,
        events: EventSink,
        provider: str,
        model: str,
    ) -> None:
        self._metrics = metrics
        self._events = events
        self._provider = (provider or "unknown")[:64]
        self._model = (model or "unknown")[:96]

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
        except Exception as telemetry_error:
            logger.warning("Unable to record retry metric", exc_info=telemetry_error)

        try:
            self._events.emit(
                ObservabilityEvent(
                    name="llm.retry",
                    attributes={
                        "provider": self._provider,
                        "model": self._model,
                        "attempt": attempt,
                        "next_attempt": attempt + 1,
                        "delay_seconds": delay_seconds,
                        "error_type": type(exc).__name__,
                        "correlation_id": get_correlation_id(),
                    },
                )
            )
        except Exception as telemetry_error:
            logger.warning("Unable to emit retry event", exc_info=telemetry_error)
