from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000, description="User message")


class AgentResponse(BaseModel):
    response: str = Field(..., description="The response from the agent")
    tool_used: str = Field(..., description="The tool used by the agent, if any")


# Pydantic models for the agent API request and response
# Benefits:
# automatic validation
# OpenAPI documentation
# typed contracts
# Easy integration with FastAPI and other frameworks
