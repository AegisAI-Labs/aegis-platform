from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from aegis_platform.agent.service import AgentService
from aegis_platform.api.models import AgentRequest, AgentResponse
from aegis_platform.config.settings import settings
from aegis_platform.llm.factory import LLMProviderFactory

app = FastAPI(title="Aegis AI Platform")

llm_provider = LLMProviderFactory.create_llm_provider(settings)

agent_service = AgentService(
    llm_provider=llm_provider,
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


@app.post("/agent/stream", tags=["Agent"])
async def stream_agent(request: AgentRequest):
    return StreamingResponse(agent_service.stream(request.message), media_type="text/plain")


# To run the FastAPI server in development mode:
# uvicorn services.api.main:app --reload
