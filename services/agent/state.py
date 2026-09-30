from typing import TypedDict


class AgentState(TypedDict):
    user_input: str
    response: str
    tool_used: str
