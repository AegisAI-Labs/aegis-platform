# ADR-008: Provider Decorator Boundary for Observability

- **Status:** Accepted
- **Date:** 2026-11-02 (accepted for Day 8); amended 2026-11-20 (Day 14 telemetry semantics)
- **Related:** RFC-008, ADR-007, SRS FR-012 / NFR-007

## Context

AEGIS has multiple LLM providers and a shared retry mechanism.

Observability must remain provider-independent and must measure the total logical request duration, including retries and backoff.

## Decision

Compose the provider stack in the following order:

`ObservedLLMProvider(RetryingLLMProvider(ConcreteProvider))`

The observer measures the complete logical operation.

The retry wrapper reports each scheduled retry using an optional callback, for both `generate` and `stream`.

Metrics and events are written through vendor-neutral sink interfaces.

## Telemetry semantics (Day 14 amendment)

Every logical request records exactly one `llm_requests_total` and one `llm_request_duration_seconds`
point, whatever the outcome. Retries never add to `llm_requests_total`.

| Outcome | Extra metrics | Event |
|---|---|---|
| Success (`generate`) | `llm_input_tokens_total`, `llm_output_tokens_total` when usage is reported | `llm.request.completed` |
| Stream completed (including empty streams) | `llm_time_to_first_token_seconds` at the first non-empty chunk | `llm.stream.completed` |
| Failure before any chunk was delivered | `llm_requests_failed_total` | `llm.request.failed` |
| Failure after at least one chunk was delivered | `llm_requests_failed_total`, `llm_stream_interruptions_total` | `llm.stream.interrupted` |
| Task cancelled (`CancelledError`) | `llm_requests_cancelled_total` | `llm.request.cancelled` with `reason=cancelled` |
| Consumer closed the stream early (`aclose()`, client disconnect) | `llm_requests_cancelled_total` | `llm.request.cancelled` with `reason=closed_early` |
| Each scheduled retry | `llm_retries_total` | `llm.retry` |

Rules:

- Cancellation is never counted as a failure and is always re-raised.
- Labels are limited to `provider` and `model`. The model label is the request model, falling back to the
  configured default model, then `"unknown"`. It is never `"None"`.
- Events carry `correlation_id` and the exception **type** only — never prompts, completions, raw exception
  messages, API keys or headers.
- Correlation IDs come from the `X-Correlation-ID` request header when it is a safe value
  (1–128 characters of `A-Z a-z 0-9 . _ -`); otherwise one is generated. The ID is echoed in the response
  and reset after the request.
- Sink failures are logged and swallowed; they never fail the LLM request.
- In-memory sinks keep a bounded window (default 10,000 points/events) and drop the oldest entries.
- Retries happen in exactly one layer: provider SDK clients are created with `max_retries=0`
  (see ADR-007).

## Consequences

### Positive

- Provider adapters remain focused on provider-specific integration.
- Logical request duration includes retry delays.
- Retry counts are measured independently, including for streams.
- Future telemetry backends (Prometheus, OpenTelemetry — roadmap W9/W10) can be added without changing
  provider adapters.

### Trade-offs

- The observer must correctly handle async generators and cancellation.
- Retry telemetry must be best-effort.
- The in-memory sinks are intended for development and tests, not durable production telemetry.
- Streaming token usage is not recorded: the stream interface yields strings only.
