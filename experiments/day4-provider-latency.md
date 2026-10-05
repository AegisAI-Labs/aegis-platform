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

Real LLM generate and provider-level streaming (including time to first chunk)
are deferred until a usable API key is configured. No live-provider latency or
network behavior was measured in this run.

The experiment does not attempt to benchmark LLM providers against each
other. Provider selection and model quality are outside the scope of this
experiment.

### Engineering Question

Does introducing the provider abstraction create meaningful application
overhead compared with the underlying model/provider latency?

### Hypothesis

The Aegis provider abstraction should add negligible latency compared with
the network and model-inference latency of a real LLM provider. This comparison
remains untested until the real-provider scenario is run.

The abstraction should therefore allow provider independence without
introducing a significant performance penalty.

## 2. Architecture Under Test

The local baseline measured the calculator route and the fake-provider route
through AgentService and LangGraph. The following real-provider path is planned
but was not exercised in this run:

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
OpenAIProvider
   │
   ▼
Provider SDK
   │
   ▼
External LLM API

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
| Aegis commit | 9c19d56db167d3037f0a7c30538767f7c76e1fe0 |
| Provider SDK | openai 3.24.0 |
| LLM provider | OpenAI selected; real provider not invoked |
| Model | gpt-4o-mini configured; not used in local scenarios |
| Network | Not used (local-only run) |
| Test date | 2026-10-04 (local scenarios only) |

## 4. Methodology

Each local scenario was executed 10 times using the same input. Calculator,
fake-provider, and real generate measurements use AgentService and LangGraph;
FastAPI transport is not included. Real stream measurements call the provider
directly and are not comparable to API streaming latency. Real-provider
measurements were skipped because the configured API key was not usable.

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

### Scenario C — Real LLM generate()

Deferred to later implementation with a usable API key.

Execution path:

Experiment script
 ↓
AgentService
 ↓
LangGraph
 ↓
LLMProvider
 ↓
OpenAIProvider
 ↓
OpenAI API
Purpose

Establish the end-to-end latency of the real provider integration.

Record, where practical:

Request start
Provider invocation start
Provider response received
Final application response returned

### Scenario D — Real LLM stream()

Deferred to later implementation with a usable API key.

This scenario measures OpenAIProvider.stream() directly. The current
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

The local-only run completed 10 iterations per measured scenario. Values below
are the rounded figures printed by the experiment script.

### Calculator

| Metric | Result |
|---|---:|
| Iterations | 10 |
| Min | 0.0017 s |
| Median | 0.0025 s |
| Mean | 0.0027 s |
| P95 | 0.0043 s |
| Max | 0.0046 s |

### Fake LLM

| Metric | Result |
|---|---:|
| Iterations | 10 |
| Min | 0.0009 s |
| Median | 0.0015 s |
| Mean | 0.0015 s |
| P95 | 0.0023 s |
| Max | 0.0023 s |

### Real LLM — Generate

Not run; a usable API key will be configured in a later implementation phase.

### Real LLM — Stream

Not run; provider-level streaming metrics are deferred with the real LLM run.

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

Future real-provider results should be treated as an observed baseline rather
than a universal provider performance guarantee. This run has no such results.

## 10. Failure Observations

| Failure | Observed? | Aegis Error |
|---|---|---|
| Timeout | Not tested | ProviderTimeout |
| Rate limit | Not tested | ProviderRateLimited |
| Authentication | Not tested | ProviderAuthenticationError |
| Unavailable | Not tested | ProviderUnavailable |
| Request failure | Not tested | ProviderRequestError |

## 11. Engineering Findings

The calculator and fake-provider paths completed locally through AgentService
and LangGraph. The run cannot establish provider-abstraction overhead relative
to a real LLM, network latency, or streaming TTFC because no live request was
made. Provider-level streaming would not establish API time-to-first-content
with the current service.

## 12. Engineering Decision

### Decision

Keep the real-provider generate and stream scenarios for a later implementation
with a usable API key. Do not treat the local-only results as evidence for or
against real-provider performance.

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
- No real-provider or network measurements
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
