import asyncio
from collections.abc import AsyncGenerator

from aegis_platform.agent.graph import app_graph
from aegis_platform.llm.base import LLMProvider
from aegis_platform.llm.models import LLMRequest


class AgentService:
    """
    Service layer between FastAPI and LangGraph.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
    ) -> None:
        self._llm_provider = llm_provider

    async def run(self, message: str) -> dict:
        result = await app_graph.ainvoke(
            {
                "user_input": message,
                "response": "",
                "tool_used": "",
            },
            config={"configurable": {"llm_provider": self._llm_provider}},
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

    async def stream_with_llm_response(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Yield model output as it arrives."""
        async for word in self._llm_provider.stream(request):
            yield word
