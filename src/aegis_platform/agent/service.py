import asyncio
from collections.abc import AsyncGenerator

from aegis_platform.agent.graph import app_graph
from aegis_platform.llm.base import LLMProvider


class AgentService:
    """
    Service layer between FastAPI and LangGraph.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
    ) -> None:
        self.llm_provider = llm_provider

    async def run(self, message: str) -> dict:
        result = await app_graph.ainvoke(
            {
                "user_input": message,
                "response": "",
                "tool_used": "",
            },
            config={"configurable": {"llm_provider": self.llm_provider}},
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
