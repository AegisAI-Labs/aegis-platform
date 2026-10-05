import pytest

from aegis_platform.tools.registry import ToolRegistry


def test_registry_register_get_has_and_list() -> None:
    registry = ToolRegistry()

    def tool(expression: str) -> str:
        return expression

    assert not registry.has("calculator")
    registry.register("calculator", tool)

    assert registry.has("calculator")
    assert registry.get("calculator") is tool
    assert registry.list_tools() == ["calculator"]


def test_registry_rejects_missing_tool() -> None:
    with pytest.raises(ValueError, match="Tool not registered: missing"):
        ToolRegistry().get("missing")
