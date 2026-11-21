# ADR-011: Real Agent Streaming Through LangGraph and API Error Mapping

- **Status:** Accepted
- **Date:** 2026-11-20 (Day 15)
- **Related:** ADR-003 (LangGraph), ADR-005 (streaming and error boundaries), ADR-007, ADR-008, RFC-005

## Context

`POST /agent/stream` did not stream. `AgentService.stream()` ran the whole graph with `generate()`, then
replayed the finished answer word by word with a fixed 50 ms sleep. Consequences:

- Time to first token equalled total generation time; `llm_time_to_first_token_seconds` was never recorded
  for API traffic.
- The streaming retry rule (retry only before the first chunk) and stream-interruption telemetry were
  unreachable from the API.
- Errors had no consistent mapping: `/agent/run` returned unstructured 500s for provider failures and
  invalid calculator input; the SSE route caught only five error types, so others ended the stream with
  no `error` event.

The calculator tool also used `eval`, and keyword routing matched substrings (`"address"` matched `"add"`).

## Decision

1. **Streaming stays inside the LangGraph workflow.** `AgentService.stream()` runs
   `app_graph.astream(..., stream_mode=["custom", "values"])` with `configurable.stream = True`.
   The `llm` node calls `provider.stream()` and forwards each chunk through LangGraph's stream writer;
   the full response is still written to graph state. Tool nodes are unchanged; when a route produces no
   streamed chunks (calculator), the final response is sent as a single token.
2. **Chunks are forwarded unchanged.** No artificial word splitting, trailing spaces or sleeps.
3. **SSE contract is unchanged:** `token` → `{"text": ...}`, `done` → `{"status": "completed"}`,
   `error` → `{"message": ...}`. An additive `code` field is included in `error` events.
4. **One error classification for both routes** (`api/errors.py`):

   | Exception | HTTP | code |
   |---|---|---|
   | `ToolExecutionError` (invalid tool input) | 400 | `invalid_tool_input` |
   | `ProviderRateLimited` | 429 | `provider_rate_limited` |
   | `ProviderQuotaExhausted` | 503 | `provider_quota_exhausted` |
   | `ProviderUnavailable` | 503 | `provider_unavailable` |
   | `ProviderTimeout`, `TimeoutError` (retry deadline) | 504 | `provider_timeout` |
   | `ProviderAuthenticationError` | 502 | `provider_authentication_failed` |
   | other `LLMProviderError` | 502 | `provider_error` |
   | anything else (SSE only) | — | `internal_error` |

   `/agent/run` returns `{"error": {"code", "message", "correlation_id"}}`. Messages are fixed strings;
   provider exception text is never returned. Unexpected exceptions on `/agent/run` keep FastAPI's default
   500 handling.
5. **Cancellation is not an error.** A client disconnect cancels the graph and the provider stream; it is
   recorded as cancelled telemetry (ADR-008) and no `error` event is sent.
6. **Calculator safety:** expressions are evaluated by an AST whitelist (numbers, `+ - * / // % **`,
   unary `+ -`, parentheses) with length and exponent limits. Routing uses whole-word keywords and only
   selects the calculator when an arithmetic expression can be extracted (including simple phrases such as
   "add 2 and 3", "multiply 2 by 3"); otherwise the request goes to the LLM.

## Consequences

### Positive

- Real time-to-first-token for API clients and real TTFT metrics.
- Streaming retry and interruption semantics now apply to API traffic.
- Predictable HTTP status codes and SSE error events; no provider details leak.
- The calculator can no longer execute arbitrary Python.

### Trade-offs / compatibility

- **Token granularity changed:** clients receive provider-sized chunks instead of `"word "` tokens.
  Clients that concatenate `token.text` get the same final text without the extra trailing space.
- `/agent/run` now returns 4xx/5xx JSON for tool and provider failures instead of raising.
- Requests such as "calculate the area of a circle" now go to the LLM instead of failing in the
  calculator.
