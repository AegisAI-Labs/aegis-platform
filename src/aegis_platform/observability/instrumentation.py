import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from .context import get_correlation_id
from .contracts import (
    EventSink,
    MetricPoint,
    MetricsSink,
    ObservabilityEvent,
)

logger = logging.getLogger(__name__)


class ObservedLLMProvider:
    """Decorates an existing LLMProvider with operational telemetry.

    Expected provider contract:
        async generate(request) -> LLMResponse
        stream(request) -> AsyncIterator[str]
    """

    def __init__(
        self,
        provider: Any,
        metrics: MetricsSink,
        events: EventSink,
    ) -> None:
        self._provider = provider
        self._metrics = metrics
        self._events = events

    @property
    def provider_name(self) -> str:
        return str(getattr(self._provider, "provider_name", "unknown"))

    def _record_metric(
        self,
        name: str,
        value: float,
        provider: str,
        model: str,
    ) -> None:
        try:
            self._metrics.record(
                MetricPoint(
                    name=name,
                    value=value,
                    labels={
                        "provider": provider[:64],
                        "model": model[:96],
                    },
                )
            )
        except Exception:
            # Telemetry must never break an LLM request.
            logger.warning(
                "Metrics sink failed while recording %s",
                name,
            )

    def _emit_event(
        self,
        name: str,
        attributes: dict[str, str | int | float | bool],
    ) -> None:
        try:
            self._events.emit(
                ObservabilityEvent(
                    name=name,
                    attributes=attributes,
                )
            )
        except Exception:
            logger.warning(
                "Event sink failed while emitting %s",
                name,
            )

    async def generate(self, request: Any) -> Any:
        started = time.perf_counter()

        provider = self.provider_name
        model = str(getattr(request, "model", "unknown"))
        correlation_id = get_correlation_id()

        try:
            response = await self._provider.generate(request)

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            elapsed = time.perf_counter() - started

            self._record_metric("llm_requests_total", 1, provider, model)
            self._record_metric("llm_requests_failed_total", 1, provider, model)
            self._record_metric(
                "llm_request_duration_seconds",
                elapsed,
                provider,
                model,
            )

            self._emit_event(
                "llm.request.failed",
                {
                    "provider": provider,
                    "model": model,
                    "correlation_id": correlation_id,
                    "error_type": type(exc).__name__,
                },
            )

            raise

        elapsed = time.perf_counter() - started

        response_provider = str(getattr(response, "provider", provider))
        response_model = str(getattr(response, "model", model))

        self._record_metric(
            "llm_requests_total",
            1,
            response_provider,
            response_model,
        )

        self._record_metric(
            "llm_request_duration_seconds",
            elapsed,
            response_provider,
            response_model,
        )

        usage = getattr(response, "usage", None)

        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)

            if input_tokens is not None:
                self._record_metric(
                    "llm_input_tokens_total",
                    float(input_tokens),
                    response_provider,
                    response_model,
                )

            if output_tokens is not None:
                self._record_metric(
                    "llm_output_tokens_total",
                    float(output_tokens),
                    response_provider,
                    response_model,
                )

        self._emit_event(
            "llm.request.completed",
            {
                "provider": response_provider,
                "model": response_model,
                "correlation_id": correlation_id,
                "duration_seconds": elapsed,
            },
        )

        return response

    def stream(self, request: Any) -> AsyncIterator[str]:
        async def iterator() -> AsyncIterator[str]:
            started = time.perf_counter()

            first_token_at: float | None = None
            completed = False

            provider = self.provider_name
            model = str(getattr(request, "model", "unknown"))
            correlation_id = get_correlation_id()

            try:
                async for chunk in self._provider.stream(request):
                    if chunk and first_token_at is None:
                        first_token_at = time.perf_counter()

                        self._record_metric(
                            "llm_time_to_first_token_seconds",
                            first_token_at - started,
                            provider,
                            model,
                        )

                    yield chunk

                completed = True

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                self._record_metric(
                    "llm_requests_failed_total",
                    1,
                    provider,
                    model,
                )

                self._record_metric(
                    "llm_stream_interruptions_total",
                    1,
                    provider,
                    model,
                )

                self._emit_event(
                    "llm.stream.interrupted",
                    {
                        "provider": provider,
                        "model": model,
                        "correlation_id": correlation_id,
                        "error_type": type(exc).__name__,
                    },
                )

                raise

            finally:
                elapsed = time.perf_counter() - started

                self._record_metric(
                    "llm_requests_total",
                    1,
                    provider,
                    model,
                )

                self._record_metric(
                    "llm_request_duration_seconds",
                    elapsed,
                    provider,
                    model,
                )

                if completed:
                    self._emit_event(
                        "llm.stream.completed",
                        {
                            "provider": provider,
                            "model": model,
                            "correlation_id": correlation_id,
                            "duration_seconds": elapsed,
                        },
                    )

        return iterator()


# Notes:
# 1. time.perf_counter() measures elapsed duration rather than wall-clock time.
# 2. The observer measures the entire retry operation because it wraps the retry provider.
# 3. TTFT is measured at the first non-empty chunk.
# 4. Streaming output is yielded unchanged.
# 5. Exceptions are re-raised; telemetry does not hide provider failures.
# 6. Token usage is recorded only when the provider supplies it.
# 7. No prompts, completions, or raw exception messages are included in events.
# Streaming cancellation: Cancellation is propagated, not converted into a provider failure. The finally block still records the operation count and duration.
