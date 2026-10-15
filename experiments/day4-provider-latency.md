# Day 4 — LLM Provider Latency Experiment

## 1. Experiment Overview

### Objective

Measure the latency characteristics of the Aegis LLM provider abstraction
and compare the execution paths used during Day 4 development.

The experiment is intended to establish a baseline for future reliability,
performance, provider-routing, and optimization work.

### Scope

This Day 4 run evaluates:

- Calculator/tool execution baseline
- Fake LLM provider execution
- NVIDIA provider generate and provider-level streaming

The live scenarios use the configured NVIDIA provider. Generate requests run
through AgentService and LangGraph; streaming calls the provider directly.

The experiment does not attempt to benchmark LLM providers against each
other. Provider selection and model quality are outside the scope of this
experiment.

### Engineering Question

Does introducing the provider abstraction create meaningful application
overhead compared with the underlying model/provider latency?

### Hypothesis

The Aegis provider abstraction should add negligible latency compared with
the network and model-inference latency of a real LLM provider. This run
provides a live-provider baseline, but does not isolate abstraction overhead
from network and model-inference latency.

The abstraction should therefore allow provider independence without
introducing a significant performance penalty.

## 2. Architecture Under Test

The local baseline measured the calculator route and the fake-provider route
through AgentService and LangGraph. The live generate path also runs through
AgentService and LangGraph; provider streaming is measured directly:

Experiment script
   │
   ▼
AgentService
   │
   ▼
LangGraph
   │
   ▼
LLMProvider
   │
   ▼
NvidiaProvider
   │
   ▼
Provider SDK
   │
   ▼
NVIDIA API

For the fake-provider test:

AgentService
     │
     ▼
LangGraph
     │
     ▼
LLMProvider
     │
     ▼
FakeLLMProvider

## 3. Test Environment

| Property | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.12.14 |
| Aegis commit | Not recorded |
| Provider SDK | OpenAI-compatible SDK; version not recorded |
| LLM provider | NVIDIA (`NvidiaProvider`) |
| Model | `openai/gpt-oss-20b` (`NVIDIA_MODEL`) |
| Network | Live NVIDIA API requests for generate and stream |
| Test date | 2026-10-14 |

The script header prints `LLM_MODEL` (`gpt-4o-mini`), but the NVIDIA factory
selects `NVIDIA_MODEL` for requests. The measured NVIDIA model was therefore
`openai/gpt-oss-20b`.

## 4. Methodology

Each scenario was executed 10 times using the same input. Calculator,
fake-provider, and real generate measurements use AgentService and LangGraph;
FastAPI transport is not included. Real stream measurements call the provider
directly and are not comparable to API streaming latency. The live generate
and stream requests used the configured NVIDIA provider.

The first execution is included in the reported sample and may include
initialization or connection establishment effects.

Measurements are collected using a monotonic clock.

For each scenario, record:

- Number of iterations
- Minimum latency
- Maximum latency
- Mean latency
- Median latency
- P95 latency
- P99 latency where the sample size is sufficient

Why median and P95?

Because average latency alone can hide bad outliers.

For example:

Mean:   1.2 sec
Median: 0.8 sec
P95:    2.9 sec

That tells you that most requests are fast, but some requests are much slower.

## 5. Experimental Scenarios:

### Scenario A — Calculator

Execution path:

Experiment script
 ↓
AgentService
 ↓
LangGraph
 ↓
Calculator

Example input:

calculate 25 * 17
Purpose

Establish a local application baseline without LLM network or inference
latency.

This provides a reference point for the cost of Aegis request handling and
tool execution.

### Scenario B — Fake LLM Provider

Execution path:

Experiment script
 ↓
AgentService
 ↓
LangGraph
 ↓
LLMProvider
 ↓
FakeLLMProvider
Purpose

Measure the Aegis orchestration and provider-abstraction path without
external network or model latency.

This provides a useful approximation of the latency introduced by:

AgentService
LangGraph
Provider abstraction
Provider invocation
Response handling

### Scenario C — NVIDIA LLM generate()

Execution path:

Experiment script
 ↓
AgentService
 ↓
LangGraph
 ↓
LLMProvider
 ↓
NvidiaProvider
 ↓
NVIDIA API
Purpose

Establish the end-to-end latency of the real provider integration.

Record, where practical:

Request start
Provider invocation start
Provider response received
Final application response returned

### Scenario D — NVIDIA LLM stream()

This scenario measures NvidiaProvider.stream() directly. The current
AgentService.stream() and /agent/stream endpoint still simulate streaming by
splitting a completed generate() response into words, so these provider-level
TTFC results must not be interpreted as API time-to-first-content.

Execution path:

Request
   │
   ├── T0 = request started
   │
   ├── T1 = provider request started
   │
   ├── TTFС = first chunk received
   │
   ├── T2 = subsequent chunks
   │
   └── T3 = final chunk received

The primary streaming metrics are:

Time to First Chunk
TTFC = first_chunk_timestamp - request_start_timestamp
Total Streaming Latency
Total = final_chunk_timestamp - request_start_timestamp

Streaming performance should therefore be evaluated independently from total
response latency.

## 6. Metrics

| Metric | Description |
|---|---|
| Min | Fastest observed execution |
| Max | Slowest observed execution |
| Mean | Arithmetic average |
| Median | 50th percentile |
| P95 | 95th percentile latency |
| P99 | 99th percentile latency |
| TTFC | Time to first streamed chunk |
| Total | End-to-end execution time |

## 7. Results

The run completed 10 iterations for each scenario. Values below are the
rounded figures printed by the experiment script. All 10 live generate and
stream requests completed without a reported error.

### Calculator

| Metric | Result |
|---|---:|
| Iterations | 10 |
| Min | 0.0014 s |
| Median | 0.0019 s |
| Mean | 0.0023 s |
| P95 | 0.0046 s |
| Max | 0.0062 s |

### Fake LLM

| Metric | Result |
|---|---:|
| Iterations | 10 |
| Min | 0.0009 s |
| Median | 0.0015 s |
| Mean | 0.0016 s |
| P95 | 0.0022 s |
| Max | 0.0024 s |

### NVIDIA LLM — Generate

| Metric | Result |
|---|---:|
| Iterations | 10 |
| Min | 1.2375 s |
| Median | 2.5027 s |
| Mean | 2.5527 s |
| P95 | 4.7280 s |
| Max | 5.1065 s |

### NVIDIA LLM — Stream

| Metric | Result |
|---|---:|
| Iterations | 10 |
| Min TTFC | 0.7612 s |
| Median TTFC | 1.2613 s |
| Mean TTFC | 2.1033 s |
| P95 TTFC | 4.4972 s |
| Max TTFC | 5.3514 s |
| Min Total | 1.3364 s |
| Median Total | 1.9966 s |
| Mean Total | 3.2557 s |
| P95 Total | 7.3933 s |
| Max Total | 8.1892 s |

## 8. Provider Abstraction Overhead

The experiment does not attempt to precisely isolate network latency,
provider-side processing, model inference, and local application overhead.

The FakeLLMProvider provides an approximation of AgentService and LangGraph
overhead, while the real-provider generate measurements include that same
path plus provider and network time. Neither includes FastAPI transport.

A more rigorous decomposition will be performed during future performance
and load-testing work.

## 9. Variability

Real-provider measurements are expected to vary due to factors outside
Aegis control, including:

- Network conditions
- Provider queueing
- Model availability
- Provider-side load
- Token generation characteristics
- Request size
- Response size

These NVIDIA results are an observed baseline rather than a universal provider
performance guarantee. Streaming TTFC and total duration varied substantially
across the 10 sequential samples.

## 10. Failure Observations

| Failure | Observed? | Aegis Error |
|---|---|---|
| Timeout | Not observed | ProviderTimeout |
| Rate limit | Not observed | ProviderRateLimited |
| Authentication | Not observed | ProviderAuthenticationError |
| Unavailable | Not observed | ProviderUnavailable |
| Request failure | Not observed | ProviderRequestError |

## 11. Engineering Findings

The calculator and fake-provider paths completed locally through AgentService
and LangGraph. Ten NVIDIA generate requests completed with a median latency of
2.5027 s; ten provider-level streams had a median TTFC of 1.2613 s and median
total duration of 1.9966 s. These measurements include network and provider
behavior and do not isolate Aegis abstraction overhead. Provider-level
streaming does not establish API time-to-first-content with the current
service.

## 12. Engineering Decision

### Decision

Retain the NVIDIA generate and stream scenarios as a live baseline. Do not
interpret these 10 sequential samples as a production performance guarantee
or as an isolated measurement of provider-abstraction overhead.

### Rationale

The provider abstraction offers potential benefits to evaluate alongside
the future real-provider measurements:

- Provider independence
- Testability
- Normalized error handling
- Provider replacement capability
- Streaming support
- Cleaner AgentService boundaries

## 13. Limitations

This experiment is an engineering baseline rather than a production load
test.

Limitations include:

- Small sample size
- Single development environment
- No concurrent workload
- No sustained load
- No geographic distribution
- No production network conditions
- No statistically controlled provider-side conditions

Comprehensive load and performance testing is planned for the later
performance-testing phase of Aegis.

## 14. Follow-Up Work

Future experiments should evaluate:

- Concurrent requests
- Provider throughput
- Sustained load
- P95/P99 latency under load
- Provider failover
- Retry behavior
- Rate limiting
- Circuit breakers
- Multiple LLM providers
- Different model sizes
- Token-based cost/latency relationships
- Kubernetes deployment latency
- Kafka-based asynchronous execution
