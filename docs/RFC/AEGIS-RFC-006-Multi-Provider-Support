# RFC-006: Multi-Provider Support

- **Status:** Proposed
- **Date:** 2026-10-18
- **Owner:** Aegis AI Platform
- **Related:** Existing LLM provider abstraction, NVIDIA adapter, OpenAI adapter

## 1. Summary

Establish and validate multi-provider LLM support in Aegis AI Platform. NVIDIA is the default provider, while OpenAI remains available as an optional provider. Both adapters must satisfy the existing provider-neutral contract and work through the same agent execution path.

## 2. Motivation

A production-oriented AI platform should avoid coupling its agent orchestration to one LLM vendor. Provider flexibility enables controlled experimentation, portability, and comparison of reliability, latency, response quality, streaming behavior, and usage metadata.

The NVIDIA adapter has been implemented. This RFC defines the validation and integration work required to establish that the overall platform supports multiple providers reliably.

## 3. Goals

- Use NVIDIA by default through application configuration.
- Retain OpenAI as an optional provider.
- Route provider selection through `LLMProviderFactory`.
- Preserve the existing `LLMProvider` interface and `LLMRequest`/`LLMResponse` models.
- Preserve genuine incremental streaming.
- Keep provider-specific SDK behavior and exceptions inside each adapter.
- Add automated tests for provider selection, generation, streaming, and error handling.
- Establish repeatable experiments for comparing provider behavior.
- Record real experimental results without fabricating performance or quality claims.

## 4. Non-goals

- Rewriting AgentService or LangGraph without evidence that changes are necessary.
- Implementing automatic failover or provider load balancing.
- Guaranteeing identical responses across different models.
- Adding RAG, persistent memory, Kafka, or unrelated roadmap features as part of this RFC.
- Claiming latency, cost, or quality improvements before measurement.

## 5. Proposed design

The application uses the following execution path:

API → AgentService → LangGraph → LLMProviderFactory → selected provider adapter.

The factory selects the provider using application settings. NVIDIA is the default. OpenAI is selected explicitly through configuration.

Each adapter implements the existing provider contract:

- `generate(request) -> LLMResponse`
- `stream(request) -> AsyncIterator[str]`

The agent service and graph should remain unaware of provider-specific SDK types.

Model IDs, API keys, and timeouts remain configuration-driven. API keys must not be logged or committed.

## 6. Experiment design

Use a repeatable set of prompts for both providers where credentials and model access are available.

Measure and record:

- Provider and exact model ID.
- Successful and failed request counts.
- End-to-end latency.
- Time to first streamed content, where measurable.
- Total streaming duration.
- Streaming completion and errors.
- Response quality against a fixed evaluation rubric.
- Input, output, and total token usage when returned by the provider.
- Estimated cost only when a documented pricing source and applicable rates are available.

Record environment details, test conditions, sample counts, and limitations. Do not compare results from materially different conditions as if they were equivalent.

## 7. Reliability and security

- Map provider SDK errors into Aegis provider-neutral exceptions.
- Do not expose API keys or sensitive request details in errors or logs.
- Preserve cancellation behavior and avoid swallowing unexpected programming errors.
- Test missing credentials and unsupported provider configuration.
- Keep live API tests optional for CI; use mocks for deterministic unit tests.

## 8. Acceptance criteria

1. NVIDIA is selected by default when no provider override is supplied.
2. OpenAI can be selected without changing agent orchestration code.
3. Both providers satisfy the existing interface.
4. Automated tests cover factory selection and missing credentials.
5. Generation tests verify provider-neutral response mapping.
6. Streaming tests verify incremental chunks and error propagation.
7. Existing agent, graph, calculator, and API tests continue to pass.
8. Static type checking and CI pass.
9. A repeatable comparison experiment and results template are committed.
10. Actual results, limitations, and model IDs are documented.

## 9. Risks and mitigations

- **Model capability differences:** validate supported parameters and message roles for each selected model.
- **Provider outages or rate limits:** test normalized error handling; defer automatic failover to a future decision.
- **Incomparable latency measurements:** use consistent prompts, warm-up rules, sample counts, and network conditions.
- **Incomplete usage data:** record missing fields rather than estimating token counts.
- **Unverified live access:** distinguish mocked tests from live-provider validation.

## 10. Decision required

Approve the multi-provider validation scope and use the results to determine whether further provider-specific compatibility work is necessary.
