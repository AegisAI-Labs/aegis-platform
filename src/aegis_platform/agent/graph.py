from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from aegis_platform.llm.base import LLMProvider
from aegis_platform.llm.models import LLMMessage, LLMRequest
from aegis_platform.tools.calculator import calculator
from aegis_platform.tools.errors import ToolExecutionError
from aegis_platform.tools.expression import extract_expression, has_calculator_keyword

from ..tools.registry import ToolRegistry
from .state import AgentState

tool_registry = ToolRegistry()
tool_registry.register("calculator", calculator)

# Custom stream events written by graph nodes when configurable["stream"] is true (ADR-011).
STREAM_TOKEN = "token"


def route_request(state: AgentState) -> str:
    """
    Deterministically route the request to either a registered tool or the LLM.

    The calculator is selected only when a calculator keyword is present as a whole word
    AND an arithmetic expression can be extracted; everything else goes to the LLM.
    Intelligent tool selection belongs to later agent-planning work.
    """

    message = state["user_input"]

    if (
        tool_registry.has("calculator")
        and has_calculator_keyword(message)
        and extract_expression(message) is not None
    ):
        return "calculator"

    return "llm"


def calculator_node(state: AgentState) -> dict[str, str]:
    """
    Execute the calculator through the Tool Registry.
    """

    tool = tool_registry.get("calculator")

    expression = extract_expression(state["user_input"])
    if expression is None:
        raise ToolExecutionError(f"Invalid calculator expression: {state['user_input'].strip()}")

    result = tool(expression)

    return {
        "response": result,
        "tool_used": "calculator",
    }


async def llm_node(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    configurable = config.get("configurable", {})
    provider: LLMProvider = configurable["llm_provider"]

    request = LLMRequest(messages=[LLMMessage(role="user", content=state["user_input"].strip())])

    if configurable.get("stream", False):
        # Forward provider chunks as they arrive; keep the full text in graph state.
        write = get_stream_writer()
        parts: list[str] = []
        async for chunk in provider.stream(request):
            parts.append(chunk)
            write({"type": STREAM_TOKEN, "text": chunk})
        return {"response": "".join(parts), "tool_used": "llm"}

    response = await provider.generate(request)

    return {"response": response.content, "tool_used": "llm"}


graph_builder = StateGraph(AgentState)

graph_builder.add_node(
    "calculator",
    calculator_node,
)

graph_builder.add_node(
    "llm",
    llm_node,
)

graph_builder.add_conditional_edges(
    START,
    route_request,
    {
        "calculator": "calculator",
        "llm": "llm",
    },
)

graph_builder.add_edge(
    "calculator",
    END,
)

graph_builder.add_edge(
    "llm",
    END,
)


app_graph = graph_builder.compile()
