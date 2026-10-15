# Day 5 — Streaming Reliability Experiment

## Objective

Repeat the main Day 4 baselines and extend the real-provider run with streaming
reliability checks. The experiment measures calculator and fake-provider
latency locally, then optionally measures the configured provider's
`generate()` and `stream()` calls, SDK error mappings, and consumer
cancellation.

Real `generate()` runs through `AgentService` and LangGraph. Real streaming
uses the configured provider's `stream()` method directly; it does not measure
FastAPI/SSE transport or compare providers. Both OpenAI and NVIDIA adapters
are supported through `LLMProviderFactory`.

## Run

From the repository root:

```powershell
uv run python scripts/experiment_day5_streaming_reliability.py
```

This runs metadata collection, local error-mapping checks, and the calculator
and fake-provider baselines. It does not make API calls. To enable live
scenarios, set `LLM_PROVIDER` to `nvidia` or `openai` and configure the matching
credentials and model (`NVIDIA_API_KEY`/`NVIDIA_MODEL` or
`OPENAI_API_KEY`/`LLM_MODEL`), then run:

```powershell
uv run python scripts/experiment_day5_streaming_reliability.py --live
```

Local scenarios default to 10 iterations. Live scenarios default to 2
iterations independently, limiting optional API usage. Adjust the live count
for a smaller check, for example:

```powershell
uv run python scripts/experiment_day5_streaming_reliability.py --live --live-iterations 1
```

The live run makes one generate request and one stream request per live
iteration, then one additional stream request for cancellation. SDK and
LangGraph retries may increase the actual HTTP request count. A detected
exhausted-quota error stops subsequent paid scenarios. The script prints
runtime metadata and measurements; copy actual observations into the record
below.

## Environment Record

| Property | Result |
|---|---|
| Test date | 2026-10-14 |
| Python version | 3.12.14 |
| OpenAI SDK version | 3.24.0 |
| Provider configured | NVIDIA |
| Model configured | `openai/gpt-oss-20b` |
| Operating system | Windows |
| Network/environment notes | NVIDIA live calls enabled; 10 local iterations and 1 live iteration |

## Test Scenarios

1. **Calculator baseline:** Run the Day 4 calculator expression through
   `AgentService` and LangGraph for each configured iteration.
2. **Fake LLM baseline:** Run the deterministic fake provider through
   `AgentService` and LangGraph for each configured iteration.
3. **Real generate:** When `--live` is set, call `AgentService.run()` for each
  live iteration and record total latency or the classified provider error.
4. **Real stream:** When `--live` is set, consume the configured provider
  stream to completion. Record time to first non-empty chunk, total duration,
  chunk count, and any classified provider error.
5. **Error classification:** Constructed OpenAI SDK exceptions are passed
  through the configured adapter's mapper without network calls. The expected
  mappings are authentication to `ProviderAuthenticationError`, rate limit
  to `ProviderRateLimited`, timeout to `ProviderTimeout`, connection failure
  to `ProviderUnavailable`, and other `OpenAIError` instances to
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
| Iterations | 10 |
| Successful | 10 |
| Failures | 0 |
| Min | 0.0010 s |
| Median | 0.0013 s |
| Mean | 0.0018 s |
| P95 | 0.0043 s |
| Max | 0.0054 s |

### Fake LLM Baseline

| Metric | Result |
|---|---:|
| Iterations | 10 |
| Successful | 10 |
| Failures | 0 |
| Min | 0.0009 s |
| Median | 0.0012 s |
| Mean | 0.0012 s |
| P95 | 0.0015 s |
| Max | 0.0017 s |

### Real LLM Generate

| Run | Total duration | Error classification |
|---|---:|---|
| 1 | 5.6931 s | None |

One live iteration completed successfully (1 successful, 0 failures). With a
single sample, the measured latency was 5.6931 s; this is an observation, not a
meaningful percentile estimate.

### Successful Stream

| Run | Time to first chunk | Total streaming duration | Chunks received | Error classification |
|---|---:|---:|---:|---|
| 1 | 0.8389 s | 1.9371 s | 91 | None |

One live stream completed successfully (1 successful, 0 failures). The single
sample had 0.8389 s TTFC, 1.9371 s total duration, and 91 content chunks; these
values are observations, not meaningful percentile estimates.

If the stream fails before yielding content, TTFC is not measured; record the
total duration, zero chunks, and reported error classification.

### Error Classification

| Scenario | SDK exception | Expected Aegis exception | Observed result |
|---|---|---|---|
| Authentication | `AuthenticationError` | `ProviderAuthenticationError` | PASS |
| Rate limit | `RateLimitError` | `ProviderRateLimited` | PASS |
| Exhausted quota | `RateLimitError` with `credit_balance_exhausted` | Quota exhausted; stop further paid scenarios | PASS: quota exhausted (`credit_balance_exhausted`) |
| Timeout | `APITimeoutError` | `ProviderTimeout` | PASS |
| Connection | `APIConnectionError` | `ProviderUnavailable` | PASS |
| Other SDK error | `OpenAIError` | `ProviderRequestError` | PASS |
| Live request failures | NVIDIA API | Aegis provider errors | None observed in this run |

The mapper rows test the configured adapter with synthetic SDK exceptions;
they do not prove how the remote service responds under real outage or quota
conditions. The exhausted-quota row additionally checks the SDK error code
used to avoid repeating paid requests after the account reports no usable
credits.

### Cancellation

| Measurement | Result |
|---|---|
| First chunk received | Yes; TTFC 1.2543 s |
| Chunks received before cancellation | 1 |
| Cancellation propagated to caller task | Yes |
| Cancellation-to-local-task-exit | 0.0007 s |
| Remote generation confirmed stopped | Not observable from this client |
| Error classification, if any | None |

## Known Limitations

- Live-provider measurements depend on network conditions, service load, account limits, and
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
