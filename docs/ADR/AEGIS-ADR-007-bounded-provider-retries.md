# ADR-007 — Bounded Provider Retries

- **Status:** Proposed — mark Accepted only after review
- **Date:** 2026-10-28

## Context
Aegis supports multiple LLM providers behind a common protocol. Transient failures should be recoverable without unbounded latency or duplicated streamed output.

## Decision
Adopt a provider-neutral retry decorator that wraps the existing `LLMProvider`, retries only an explicit allowlist of normalized transient exceptions, uses bounded exponential backoff with jitter, enforces a total deadline, propagates cancellation, retries streaming only before the first chunk, and emits metadata-only logs. Compose it exactly once. Cross-provider failover is out of scope.

## Alternatives
- **Retries in each adapter:** rejected because policy is duplicated.
- **Retries in graph/service:** rejected because it mixes orchestration and reliability and risks nested retries.
- **No retries:** remains a possible configuration, not the selected default.
- **Automatic provider failover:** out of scope pending a separate decision about latency, cost, model equivalence, and data handling.

## Consequences
Benefits: consistent policy, testability, improved diagnosis.
Costs: added latency/provider charges, dependence on accurate exception mappings, and no transparent recovery after a stream has emitted output.

## Validation
Pass deterministic tests, existing graph/service tests, static checks, and a documented failure-injection experiment.
