# Day 5 — Streaming Reliability Experiment

## Objective

Repeat the main Day 4 baselines and extend the real-provider run with streaming
reliability checks. The experiment measures calculator and fake-provider
latency locally, then optionally measures real OpenAI `generate()` and
`stream()` calls, SDK error mappings, and consumer cancellation.

Real `generate()` runs through `AgentService` and LangGraph. Real streaming
uses `OpenAIProvider.stream()` directly, as in Day 4; it does not measure
FastAPI/SSE transport or compare providers.

## Run

From the repository root:

```powershell
uv run python scripts/experiment_day5_streaming_reliability.py
```

This runs metadata collection, local error-mapping checks, and the calculator
and fake-provider baselines. It does not call the OpenAI API. To enable the
real-provider scenarios, configure `OPENAI_API_KEY` and `LLM_MODEL`, then run:

```powershell
uv run python scripts/experiment_day5_streaming_reliability.py --live
```

The default is 10 iterations, matching Day 4. Set a smaller count for a
cost-conscious live check, for example:

```powershell
uv run python scripts/experiment_day5_streaming_reliability.py --live --iterations 2
```

The live run makes one generate request and one stream request per iteration,
then one additional stream request for cancellation. SDK and LangGraph retries
may increase the actual HTTP request count. A detected exhausted-quota error
stops subsequent paid scenarios. The script prints runtime metadata and
measurements; copy actual observations into the record below.

## Environment Record

| Property | Result |
|---|---|
| Test date | Pending live run |
| Python version | 3.12.14 (offline harness validation) |
| OpenAI SDK version | 3.24.0 (offline harness validation) |
| Model used | `gpt-4o-mini` (configured model; offline harness validation) |
| Operating system | Windows |
| Network/environment notes | Offline validation only; live API scenarios not run |

## Test Scenarios

1. **Calculator baseline:** Run the Day 4 calculator expression through
   `AgentService` and LangGraph for each configured iteration.
2. **Fake LLM baseline:** Run the deterministic fake provider through
   `AgentService` and LangGraph for each configured iteration.
3. **Real generate:** When `--live` is set, call `AgentService.run()` for each
   iteration and record total latency or the classified provider error.
4. **Real stream:** When `--live` is set, consume the provider stream to
   completion. Record time to first non-empty chunk, total duration, chunk
   count, and any classified provider error.
5. **Error classification:** Constructed OpenAI SDK exceptions are passed
   through the adapter mapper without network calls. The expected mappings are
   authentication to `ProviderAuthenticationError`, rate limit to
   `ProviderRateLimited`, timeout to `ProviderTimeout`, connection failure to
   `ProviderUnavailable`, and other `OpenAIError` instances to
   `ProviderRequestError`.
6. **Cancellation:** A separate live stream is consumed until its first
   non-empty chunk, then the consumer task is cancelled. The script records
   chunks received before cancellation, whether cancellation propagates, and
   the time from cancellation request to local task exit. This runs only when
   live scenarios are enabled and quota has not been exhausted.

## Results

### Calculator Baseline

| Metric | Result |
|---|---:|
| Iterations | Pending |
| Min | Pending |
| Median | Pending |
| Mean | Pending |
| P95 | Pending |
| Max | Pending |

### Fake LLM Baseline

| Metric | Result |
|---|---:|
| Iterations | Pending |
| Min | Pending |
| Median | Pending |
| Mean | Pending |
| P95 | Pending |
| Max | Pending |

### Real LLM Generate

| Run | Total duration | Error classification |
|---|---:|---|
| 1 | Pending | Pending |
| 2 | Pending | Pending |
| 3 | Pending | Pending |
| 4 | Pending | Pending |
| 5 | Pending | Pending |
| 6 | Pending | Pending |
| 7 | Pending | Pending |
| 8 | Pending | Pending |
| 9 | Pending | Pending |
| 10 | Pending | Pending |

Record successful-run summary statistics and failure count from console output.
Rows beyond the selected iteration count are not applicable. Live results are
currently not measured.

### Successful Stream

| Run | Time to first chunk | Total streaming duration | Chunks received | Error classification |
|---|---:|---:|---:|---|
| 1 | Pending | Pending | Pending | Pending |
| 2 | Pending | Pending | Pending | Pending |
| 3 | Pending | Pending | Pending | Pending |
| 4 | Pending | Pending | Pending | Pending |
| 5 | Pending | Pending | Pending | Pending |
| 6 | Pending | Pending | Pending | Pending |
| 7 | Pending | Pending | Pending | Pending |
| 8 | Pending | Pending | Pending | Pending |
| 9 | Pending | Pending | Pending | Pending |
| 10 | Pending | Pending | Pending | Pending |

Record TTFC, total-duration summary statistics, failure count, and chunk-count
range from console output. Rows beyond the selected iteration count are not
applicable.

If the stream fails before yielding content, TTFC is not measured; record the
total duration, zero chunks, and reported error classification.

### Error Classification

| Scenario | SDK exception | Expected Aegis exception | Observed result |
|---|---|---|---|
| Authentication | `AuthenticationError` | `ProviderAuthenticationError` | Pending |
| Rate limit | `RateLimitError` | `ProviderRateLimited` | Pending |
| Exhausted quota | `RateLimitError` with `credit_balance_exhausted` | Quota exhausted; stop further paid scenarios | Pending |
| Timeout | `APITimeoutError` | `ProviderTimeout` | Pending |
| Connection | `APIConnectionError` | `ProviderUnavailable` | Pending |
| Other SDK error | `OpenAIError` | `ProviderRequestError` | Pending |
| Live request failures | As observed | As observed | Pending |

The mapper rows test behavior with synthetic SDK exceptions; they do not prove
how the remote service responds under real outage or quota conditions. The
exhausted-quota row additionally checks the SDK error code used to avoid
repeating paid requests after the account reports no usable credits.

### Cancellation

| Measurement | Result |
|---|---|
| First chunk received | Pending |
| Chunks received before cancellation | Pending |
| Cancellation propagated to caller task | Pending |
| Cancellation-to-local-task-exit | Pending |
| Remote generation confirmed stopped | Not observable from this client |
| Error classification, if any | Pending |

## Known Limitations

- Live results depend on network conditions, service load, account limits, and
  model behavior; a small sequential run is not a production reliability or
  load test.
- The calculator and fake-provider measurements are local baselines, not
  substitutes for live-provider latency. Compare them with Day 4 only when the
  prompt, iteration count, runtime, and environment are comparable.
- TTFC and total duration use a local monotonic clock around the SDK iterator.
  They include local scheduling and network/setup latency and are not server-
  side timings.
- Chunk counts are SDK content chunks, not necessarily HTTP frames or tokens.
- Cancellation demonstrates local task behavior only. It cannot confirm when
  the remote generation stops or whether usage is billed after cancellation.
- Cancellation is skipped if the live stream fails or yields no non-empty
  chunk before the timeout.
- Synthetic error mapping verifies the adapter's class mapping, not live
  authentication, rate-limit, timeout, or connection-failure handling end to
  end.
- LangGraph and the OpenAI SDK may retry selected failures, so one logical
  generate iteration can issue more than one HTTP request.
- The experiment does not test retries, reconnect/resume, concurrent streams,
  backpressure, or FastAPI Server-Sent Events delivery.
