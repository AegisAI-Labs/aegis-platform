# RFC-7 — Provider Reliability and Failure Handling

- **Status:** Proposed
- **Date:** 2026-10-28
- **Scope:** NVIDIA and OpenAI provider calls
- **Related:** Existing SRS, C4, ADR, EDI

## Problem
Transient provider failures need bounded recovery. Unbounded or duplicated retries increase latency/cost, hide intermittent failures, and can duplicate streamed output.

## Goals
1. Bound attempts and total elapsed time.
2. Retry only explicitly classified transient exceptions.
3. Do not retry authentication or invalid-request failures.
4. Keep SDK error normalization inside provider adapters.
5. Retry streams only before the first chunk is delivered.
6. Log retry metadata without secrets or full prompts.
7. Test without live API calls.

## Non-goals
Cross-provider failover, circuit breakers, Redis caching, distributed rate limiting, or exactly-once provider execution.

## Proposed design
A provider-neutral `RetryingLLMProvider` decorates the common `LLMProvider` protocol. Compose it once at the provider factory/composition root. Provider adapters map SDK exceptions to existing normalized exceptions.

## Policy
- Default maximum attempts: 3 total (initial + up to 2 retries).
- Default overall deadline: 30 seconds.
- Exponential backoff with jitter, capped by a maximum delay.
- Retry only explicitly injected transient exception classes and built-in `TimeoutError`.
- Do not include authentication or invalid-request errors in the retry allowlist.
- Cancellation is propagated, not retried.
- Do not start another attempt after the deadline.
- Do not claim `Retry-After` support unless implemented and tested.

## Streaming
Retry only if failure occurs before any chunk is yielded. After partial output, propagate the error; do not restart the stream. API-level behavior for a mid-stream failure must be documented separately.

## Observability
Record provider, attempt, normalized error type, delay, and final attempt count. Never log API keys, authorization headers, or full prompts/completions by default.

## Acceptance
Attempts and elapsed time are bounded; permanent failures are not retried; streams are not restarted after output; cancellation propagates; tests require no credentials; actual experiment results are recorded.
