from aegis_platform.tools.calculator import calculator
from aegis_platform.tools.registry import ToolRegistry

tool_registry = ToolRegistry()

tool_registry.register(
    "calculator",
    calculator,
)
