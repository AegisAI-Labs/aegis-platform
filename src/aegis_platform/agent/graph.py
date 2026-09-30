from langgraph.graph import END, START, StateGraph

from .registry import TOOL_REGISTRY
from .state import AgentState


def agent_node(state: AgentState):
    user_input = state["user_input"].strip()

    # Simple Calculation using the calculator tool
    if user_input.startswith("calc:"):
        expression = user_input[len("calc:") :].strip()
        result = TOOL_REGISTRY["calculator"](expression)
        return {"response": f"Calculation result: {result}", "tool_used": "calculator"}

    return {"response": f"Aegis received your input: {user_input}", "tool_used": "none"}


graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)

graph.add_edge(START, "agent")
graph.add_edge("agent", END)

app_graph = graph.compile()
