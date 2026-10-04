from collections.abc import Callable
from typing import Any

from .calculator import calculator

TOOL_REGISTRY = {"calculator": calculator}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        tool: Callable[..., Any],
    ) -> None:
        self._tools[name] = tool

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ValueError(f"Tool not registered: {name}") from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())
