# RFC-004: LLM Observability and Operational Monitoring

- **Status:** Proposed
- **Owner:** AEGIS Platform
- **Date:** 2026-11-02
- **Scope:** Provider-agnostic LLM observability

## 1. Problem Statement

AEGIS supports multiple LLM providers through a provider abstraction and retry mechanism.

However, provider integration and reliability alone do not provide sufficient operational visibility.

The platform must be able to answer:

- How many LLM requests are being executed?
- How many requests fail?
- How long does each logical request take?
- How long does a streaming request take to produce its first token?
- How many tokens are consumed?
- How often are retries scheduled?
- How frequently are streams interrupted?
- Which provider and model are responsible for the observed behavior?

## 2. Goals

1. Introduce a provider-independent observability layer.
2. Capture logical request metrics.
3. Capture generation and streaming latency.
4. Capture token usage when returned by the provider.
5. Capture retry events without changing retry semantics.
6. Provide correlation IDs for structured events.
7. Prevent sensitive data from entering telemetry.
8. Support future Prometheus and OpenTelemetry integrations.

## 3. Non-Goals

- Implementing model pricing.
- Calculating monetary LLM costs.
- Implementing budgets and quotas.
- Implementing AI FinOps.
- Building a production monitoring dashboard.
- Persisting telemetry in a production database.

## 4. Proposed Architecture

The provider stack will be:

`ObservedLLMProvider(RetryingLLMProvider(ConcreteProvider))`

The observability layer wraps the retry layer.

This allows the platform to measure logical request duration, including retry attempts and backoff delays.

Retry attempts will be measured independently through a callback.

## 5. Metrics

| Metric | Description |
|---|---|
| llm_requests_total | Total logical requests |
| llm_requests_failed_total | Failed logical requests |
| llm_request_duration_seconds | End-to-end request duration |
| llm_retries_total | Scheduled retry attempts |
| llm_time_to_first_token_seconds | Streaming time to first non-empty chunk |
| llm_input_tokens_total | Input tokens when available |
| llm_output_tokens_total | Output tokens when available |
| llm_stream_interruptions_total | Interrupted streaming requests |

## 6. Structured Events

Events include:

- llm.request.completed
- llm.request.failed
- llm.retry
- llm.stream.completed
- llm.stream.interrupted

Events should include provider, model, correlation ID, and relevant operational attributes.

## 7. Privacy Requirements

Never emit:

- Prompt content
- Completion content
- API keys
- Authorization headers
- User-identifying information
- Raw exception messages

Metric labels must remain bounded. Provider and model are allowed dimensions. Request IDs and correlation IDs must not become metric labels.

## 8. Acceptance Criteria

- Generation success and failure are measured.
- Streaming TTFT is measured.
- Stream interruption is measured.
- Token usage is recorded when available.
- Retry callbacks do not affect retry behavior.
- Observability sink failures do not fail LLM requests.
- Cancellation is propagated.
- Tests cover successful and failed operations.
