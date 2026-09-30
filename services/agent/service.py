import asyncio
from collections.abc import AsyncGenerator

from services.agent.graph import app_graph


class AgentService:
    """
    Thin service layer between the API and LangGraph.

    Future versions will delegate to provider-specific LLM adapters.
    """

    async def run(self, message: str) -> dict:
        result = app_graph.invoke(
            {
                "user_input": message,
                "response": "",
                "tool_used": "",
            }
        )

        return {
            "response": result["response"],
            "tool_used": result["tool_used"],
        }

    async def stream(self, message: str) -> AsyncGenerator[str, None]:
        result = await self.run(message)

        for word in result["response"].split():
            yield f"{word} "
            await asyncio.sleep(0.05)


agent_service = AgentService()
