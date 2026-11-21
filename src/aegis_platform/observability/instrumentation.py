"""Provider decorator that records logical-request telemetry (see ADR-008).

Every logical request records exactly one ``llm_requests_total`` and one
``llm_request_duration_seconds`` point, whatever the outcome:

* success          -> ``llm.request.completed`` / ``llm.stream.completed``
* failure          -> ``llm_requests_failed_total`` + ``llm.request.failed``
* stream failure after output was delivered
                   -> additionally ``llm_stream_interruptions_total`` + ``llm.stream.interrupted``
* cancellation / early stream close
                   -> ``llm_requests_cancelled_total`` + ``llm.request.cancelled`` (never a failure)

Events never contain prompts, completions or raw exception messages.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from aegis_platform.llm.base import LLMProvider
from aegis_platform.llm.models import LLMRequest, LLMResponse

from .context import get_correlation_id
from .contracts import (
    EventSink,
    MetricPoint,
    MetricsSink,
    ObservabilityEvent,
)

logger = logging.getLogger(__name__)

UNKNOWN = "unknown"
_MAX_PROVIDER_LABEL = 64
_MAX_MODEL_LABEL = 96

EventAttributes = dict[str, str | int | float | bool]


class ObservedLLMProvider:
    """Decorates an existing LLMProvider with operational telemetry."""

    def __init__(
        self,
        provider: LLMProvider,
        metrics: MetricsSink,
        events: EventSink,
        *,
        default_model: str | None = None,
    ) -> None:
        self._provider = provider
        self._metrics = metrics
        self._events = events
        self._default_model = default_model or None

    @property
    def provider_name(self) -> str:
        return str(getattr(self._provider, "provider_name", UNKNOWN))

    def _resolve_model(self, request: LLMRequest) -> str:
        """Request model, then configured default; never the string ``"None"``."""
        model = getattr(request, "model", None)
        return str(model or self._default_model or UNKNOWN)

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
                        "provider": provider[:_MAX_PROVIDER_LABEL],
                        "model": model[:_MAX_MODEL_LABEL],
                    },
                )
            )
        except Exception as sink_error:
            # Telemetry must never break an LLM request.
            logger.warning(
                "Metrics sink failed while recording %s",
                name,
                exc_info=sink_error,
            )

    def _emit_event(self, name: str, attributes: EventAttributes) -> None:
        try:
            self._events.emit(
                ObservabilityEvent(
                    name=name,
                    attributes=attributes,
                )
            )
        except Exception as sink_error:
            logger.warning(
                "Event sink failed while emitting %s",
                name,
                exc_info=sink_error,
            )

    def _record_request(self, elapsed: float, provider: str, model: str) -> None:
        """The single logical-request count and duration for every outcome."""
        self._record_metric("llm_requests_total", 1, provider, model)
        self._record_metric("llm_request_duration_seconds", elapsed, provider, model)

    def _record_failure(
        self,
        exc: BaseException,
        provider: str,
        model: str,
        correlation_id: str,
        *,
        output_delivered: bool = False,
    ) -> None:
        self._record_metric("llm_requests_failed_total", 1, provider, model)

        if output_delivered:
            self._record_metric("llm_stream_interruptions_total", 1, provider, model)

        self._emit_event(
            "llm.stream.interrupted" if output_delivered else "llm.request.failed",
            {
                "provider": provider,
                "model": model,
                "correlation_id": correlation_id,
                "error_type": type(exc).__name__,
            },
        )

    def _record_cancellation(
        self,
        reason: str,
        provider: str,
        model: str,
        correlation_id: str,
    ) -> None:
        self._record_metric("llm_requests_cancelled_total", 1, provider, model)
        self._emit_event(
            "llm.request.cancelled",
            {
                "provider": provider,
                "model": model,
                "correlation_id": correlation_id,
                "reason": reason,
            },
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()

        provider = self.provider_name
        model = self._resolve_model(request)
        correlation_id = get_correlation_id()

        try:
            response = await self._provider.generate(request)

        except asyncio.CancelledError:
            self._record_request(time.perf_counter() - started, provider, model)
            self._record_cancellation("cancelled", provider, model, correlation_id)
            raise

        except Exception as exc:
            self._record_request(time.perf_counter() - started, provider, model)
            self._record_failure(exc, provider, model, correlation_id)
            raise

        elapsed = time.perf_counter() - started

        response_provider = str(getattr(response, "provider", None) or provider)
        response_model = str(getattr(response, "model", None) or model)

        self._record_request(elapsed, response_provider, response_model)

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

    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        async def iterator() -> AsyncIterator[str]:
            started = time.perf_counter()
            first_token_at: float | None = None
            delivered = False

            provider = self.provider_name
            model = self._resolve_model(request)
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

                    delivered = True
                    yield chunk

            except asyncio.CancelledError:
                self._record_request(time.perf_counter() - started, provider, model)
                self._record_cancellation("cancelled", provider, model, correlation_id)
                raise

            except GeneratorExit:
                # The consumer stopped iterating (aclose / client disconnect).
                self._record_request(time.perf_counter() - started, provider, model)
                self._record_cancellation("closed_early", provider, model, correlation_id)
                raise

            except Exception as exc:
                self._record_request(time.perf_counter() - started, provider, model)
                self._record_failure(
                    exc,
                    provider,
                    model,
                    correlation_id,
                    output_delivered=delivered,
                )
                raise

            elapsed = time.perf_counter() - started
            self._record_request(elapsed, provider, model)
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
# 6. Token usage is recorded only when the provider supplies it (not available for streams).
