import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from aegis_platform.agent.service import AgentService
from aegis_platform.api.models import AgentRequest, AgentResponse
from aegis_platform.config.settings import settings
from aegis_platform.llm.errors import (
    ProviderAuthenticationError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)
from aegis_platform.llm.factory import LLMProviderFactory
from aegis_platform.llm.reliability import RetryingLLMProvider, RetryPolicy

app = FastAPI(title="Aegis AI Platform")

llm_provider = LLMProviderFactory.create_llm_provider(settings)

retrying_llm_provider = RetryingLLMProvider(
    llm_provider,
    policy=RetryPolicy(
        max_attempts=3,
        timeout_seconds=100,  # overall deadline, separate from the provider's 30s per-call timeout
        initial_delay_seconds=1,
        max_delay_seconds=5,
        jitter=True,
    ),
    retryable_exceptions=(
        ProviderRateLimited,
        ProviderTimeout,
        ProviderUnavailable,
    ),
)

agent_service = AgentService(
    llm_provider=retrying_llm_provider,
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


@app.post("/agent/run", response_model=AgentResponse, tags=["Agent"])
async def run_agent(request: AgentRequest):
    result = await agent_service.run(request.message)
    return AgentResponse(**result)


# Server-Sent Events (SSE) helper function
def sse_event(event: str, data: dict[str, str]) -> str:
    return f"event: {event}\n" f"data: {json.dumps(data)}\n\n"


@app.post("/agent/stream")
async def stream_agent(request: AgentRequest):
    # Return a streaming response using Server-Sent Events (SSE)
    # Handles the incoming HTTP request and returns a streaming HTTP response.
    async def generate():
        # Generate response chunks incrementally
        # Produces response chunks incrementally for FastAPI to send to the client.
        try:
            async for chunk in agent_service.stream(request.message):
                yield sse_event("token", {"text": chunk})

            yield sse_event("done", {"status": "completed"})

        except (
            ProviderAuthenticationError,
            ProviderRateLimited,
            ProviderTimeout,
            ProviderUnavailable,
        ):
            # Do not expose raw provider exception details.
            yield sse_event(
                "error",
                {"message": "The AI provider could not complete the request."},
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# @app.post("/agent/stream", tags=["Agent"])
# async def stream_agent(request: AgentRequest):
#     return StreamingResponse(agent_service.stream(request.message), media_type="text/plain")


# To run the FastAPI server in development mode:
# uvicorn services.api.main:app --reload
