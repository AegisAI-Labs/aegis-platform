# ADR-003: LLM Streaming and Error Boundaries

- Status: Accepted
- Date: 2026-10-11

## Context

Aegis requires incremental LLM streaming and predictable error
handling across provider implementations.

Provider-specific SDK exceptions and stream objects should not leak
into the Agent Service.

## Decision

1. Provider adapters own SDK-specific streaming behavior.
2. The Agent Service consumes a provider-independent interface.
3. Streaming uses an asynchronous iterator.
4. Provider exceptions are translated into normalized errors.
5. Timeouts are configured centrally.
6. Automatic retries are not performed after partial output has
   been delivered.
7. Fake providers are used for deterministic tests.

## Consequences

Positive:
- Provider-independent orchestration.
- Easier automated testing.
- More predictable failure handling.
- Reduced coupling to vendor SDKs.

Trade-offs:
- Streaming failures require a defined post-start error strategy.
- Different providers may require different internal SDK adapters.
- Cancellation and timeout behavior must be tested explicitly.

## References

- RFC-005
- SRS: LLM Streaming and Reliability Requirements
