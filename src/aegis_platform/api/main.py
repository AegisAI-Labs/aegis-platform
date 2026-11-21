import json
import logging

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from aegis_platform.agent.service import AgentService
from aegis_platform.api.errors import classify_error, register_exception_handlers
from aegis_platform.api.models import AgentRequest, AgentResponse
from aegis_platform.api.rag import router as rag_router
from aegis_platform.config.settings import settings
from aegis_platform.llm.errors import (
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)
from aegis_platform.llm.factory import LLMProviderFactory
from aegis_platform.llm.reliability import RetryingLLMProvider, RetryPolicy
from aegis_platform.observability import (
    InMemoryEventSink,
    InMemoryMetricsSink,
    ObservedLLMProvider,
)
from aegis_platform.observability.middleware import CorrelationIdMiddleware
from aegis_platform.observability.retry_telemetry import (
    RetryTelemetry,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Aegis AI Platform")

# Map tool and provider failures to structured HTTP errors (ADR-011).
register_exception_handlers(app)

# Bind an X-Correlation-ID to every request (accepted from the caller or generated).
app.add_middleware(CorrelationIdMiddleware)

# Export .env to os.environ: the RAG router and the OpenAI SDK read the environment
# directly. Like Settings, the file is found from the working directory; variables
# already set in the environment take precedence.
load_dotenv(find_dotenv(usecwd=True))

# Include RAG router
app.include_router(rag_router, prefix="/rag", tags=["RAG"])

llm_provider = LLMProviderFactory.create_llm_provider(settings)

provider_name = getattr(
    llm_provider,
    "provider_name",
    "unknown",
)

model_name = settings.active_model

# Observability sinks (bounded, in-memory)
metrics = InMemoryMetricsSink(max_points=settings.observability_max_records)
events = InMemoryEventSink(max_events=settings.observability_max_records)

# Retry telemetry
retry_telemetry = RetryTelemetry(
    metrics=metrics,
    events=events,
    provider=provider_name,
    model=model_name,
)

# Configure a retry policy for the LLM provider
retrying_llm_provider = RetryingLLMProvider(
    llm_provider,
    policy=RetryPolicy(
        max_attempts=settings.llm_retry_max_attempts,
        # overall deadline, separate from the provider's per-call timeout
        timeout_seconds=settings.llm_retry_deadline_seconds,
        initial_delay_seconds=settings.llm_retry_initial_delay_seconds,
        max_delay_seconds=settings.llm_retry_max_delay_seconds,
        jitter=settings.llm_retry_jitter,
    ),
    retryable_exceptions=(
        ProviderRateLimited,
        ProviderTimeout,
        ProviderUnavailable,
    ),
    on_retry=retry_telemetry,
)

# Observed LLM provider wraps the retrying provider with observability features
observed_provider = ObservedLLMProvider(
    provider=retrying_llm_provider,
    metrics=metrics,
    events=events,
    default_model=model_name,
)

# Agent service uses the observed LLM provider, which wraps the retrying provider with observability features, for handling requests with retries
agent_service = AgentService(
    llm_provider=observed_provider,
)


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.get("/health/ready", tags=["Health"])
def readiness():
    return {"status": "ready"}


@app.get("/health/live", tags=["Health"])
def liveness():
    return {"status": "live"}


@app.get("/observability/metrics", tags=["Observability"])
def get_metrics():
    """Get all recorded metrics."""
    return {"metrics": metrics.snapshot()}


@app.get("/observability/events", tags=["Observability"])
def get_events():
    """Get all recorded events."""
    return {"events": [{"name": e.name, "attributes": e.attributes} for e in events.snapshot()]}


@app.post("/agent/run", response_model=AgentResponse, tags=["Agent"])
async def run_agent(request: AgentRequest) -> AgentResponse:
    result = await agent_service.run(request.message)
    return AgentResponse(**result)


# Server-Sent Events (SSE) helper function
def sse_event(event: str, data: dict[str, str]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/agent/stream", tags=["Agent"])
async def stream_agent(request: AgentRequest) -> StreamingResponse:
    """Stream the agent response as Server-Sent Events (ADR-011).

    Events: ``token`` {"text"}, ``done`` {"status": "completed"}, ``error`` {"message", "code"}.
    A client disconnect cancels the graph and the provider stream; no error event is sent.
    """

    async def generate():  # type: ignore[no-untyped-def]
        try:
            async for chunk in agent_service.stream(request.message):
                yield sse_event("token", {"text": chunk})

            yield sse_event("done", {"status": "completed"})

        except Exception as exc:
            # Headers are already sent, so failures are reported in-band.
            # Never expose raw provider exception details.
            info = classify_error(exc)
            if info.code == "internal_error":
                logger.exception("agent_stream_failed")
            else:
                logger.warning(
                    "agent_stream_failed",
                    extra={"code": info.code, "error_type": type(exc).__name__},
                )
            yield sse_event("error", {"message": info.message, "code": info.code})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# To run the FastAPI server in development mode (from the repository root):
# PYTHONPATH=src uvicorn aegis_platform.api.main:app --reload
