# ADR-004: Provider Decorator Boundary for Observability

- **Status:** Accepted for Day 8 implementation

## Context

AEGIS has multiple LLM providers and a shared retry mechanism.

Observability must remain provider-independent and must measure the total logical request duration, including retries and backoff.

## Decision

Compose the provider stack in the following order:

`ObservedLLMProvider(RetryingLLMProvider(ConcreteProvider))`

The observer measures the complete logical operation.

The retry wrapper reports each scheduled retry using an optional callback.

Metrics and events are written through vendor-neutral sink interfaces.

## Consequences

### Positive

- Provider adapters remain focused on provider-specific integration.
- Logical request duration includes retry delays.
- Retry counts are measured independently.
- Future telemetry backends can be added without changing provider adapters.

### Trade-offs

- The observer must correctly handle async generators and cancellation.
- Retry telemetry must be best-effort.
- The initial in-memory sinks are intended for development and tests, not durable production telemetry.
