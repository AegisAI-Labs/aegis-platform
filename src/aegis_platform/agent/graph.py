from langchain_protocol import Any
from langgraph.graph import END, START, StateGraph

from aegis_platform.llm.base import LLMProvider
from aegis_platform.llm.models import LLMMessage, LLMRequest
from aegis_platform.tools.calculator import calculator

from ..tools.registry import ToolRegistry
from .state import AgentState

tool_registry = ToolRegistry()
tool_registry.register("calculator", calculator)


def route_request(state: AgentState) -> str:
    """
    Deterministically route the request to either a registered
    tool or the LLM.

    This is intentionally simple for Day 4.
    Intelligent tool selection belongs to the later agent-planning work.
    """

    message = state["user_input"].lower()

    calculator_keywords = (
        "calculate",
        "calculator",
        "add",
        "subtract",
        "multiply",
        "divide",
    )

    if tool_registry.has("calculator") and any(
        keyword in message for keyword in calculator_keywords
    ):
        return "calculator"

    return "llm"


def calculator_node(state: AgentState) -> dict[str, Any]:
    """
    Execute the calculator through the Tool Registry.
    """

    tool = tool_registry.get("calculator")

    expression = state["user_input"]

    # A dedicated tool-input parser can be introduced later.
    expression = expression.replace("calculate", "").replace("calculator", "").strip()

    result = tool(expression)

    return {
        "response": result,
        "tool_used": "calculator",
    }


async def llm_node(state: AgentState, config) -> dict:
    provider: LLMProvider = config["configurable"]["llm_provider"]

    request = LLMRequest(messages=[LLMMessage(role="user", content=state["user_input"].strip())])

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
