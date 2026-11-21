from collections.abc import AsyncGenerator

from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamMode

from aegis_platform.agent.graph import STREAM_TOKEN, app_graph
from aegis_platform.agent.state import AgentState
from aegis_platform.llm.base import LLMProvider
from aegis_platform.llm.models import LLMRequest

# Custom events carry streamed tokens; values carry the final graph state.
_STREAM_MODES: list[StreamMode] = ["custom", "values"]


class AgentService:
    """
    Service layer between FastAPI and LangGraph.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
    ) -> None:
        self._llm_provider = llm_provider

    @staticmethod
    def _initial_state(message: str) -> AgentState:
        return {"user_input": message, "response": "", "tool_used": ""}

    def _config(self, *, stream: bool) -> RunnableConfig:
        return RunnableConfig(configurable={"llm_provider": self._llm_provider, "stream": stream})

    async def run(self, message: str) -> dict[str, str]:
        result = await app_graph.ainvoke(
            self._initial_state(message),
            config=self._config(stream=False),
        )

        return {
            "response": result["response"],
            "tool_used": result["tool_used"],
        }

    async def stream(self, message: str) -> AsyncGenerator[str, None]:
        """Yield response text as it is produced (ADR-011).

        LLM routes forward provider chunks unchanged. Routes that do not stream (tools)
        yield their final response once.
        """
        streamed_any = False
        final_response = ""

        async for mode, payload in app_graph.astream(
            self._initial_state(message),
            config=self._config(stream=True),
            stream_mode=_STREAM_MODES,
        ):
            if mode == "custom" and isinstance(payload, dict):
                if payload.get("type") == STREAM_TOKEN:
                    streamed_any = True
                    yield str(payload.get("text", ""))
            elif mode == "values" and isinstance(payload, dict):
                final_response = str(payload.get("response", ""))

        if not streamed_any and final_response:
            yield final_response

    async def stream_with_llm_response(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Yield model output as it arrives, bypassing the graph."""
        async for word in self._llm_provider.stream(request):
            yield word
