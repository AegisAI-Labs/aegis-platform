# Experiment 004 — LLM Observability Baseline

## Objective

Measure baseline LLM operational behavior across configured providers.

## Experiment Scope

- Generation latency.
- Streaming latency.
- Time to first token.
- Failure rate.
- Retry frequency.
- Token usage.

## Procedure

1. Select identical request inputs for each provider.
2. Run at least 20 generation requests per provider where practical.
3. Run at least 20 streaming requests per provider where practical.
4. Record provider and model.
5. Record success and failure counts.
6. Record median and p95 request duration.
7. Record median TTFT.
8. Record retry counts.
9. Record input/output token usage when available.
10. Document environment, concurrency, and limitations.

## Results

| Provider | Model | Mode | Requests | Success | Failures | Median Latency | P95 Latency | Median TTFT | Retries |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| | | Generate | | | | | | N/A | |
| | | Stream | | | | | | | |
