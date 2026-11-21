"""The calculator must evaluate arithmetic only (ADR-011): no eval, bounded work."""

import pytest

from aegis_platform.tools.calculator import MAX_EXPRESSION_LENGTH, calculator
from aegis_platform.tools.errors import ToolExecutionError


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2+3", "5"),
        ("7 * 9", "63"),
        ("8/2", "4.0"),
        ("7//2", "3"),
        ("7%4", "3"),
        ("2**10", "1024"),
        ("-(3 - 5) * +2", "4"),
        ("(1.5 + 2.5) * 2", "8.0"),
        ("2**-1", "0.5"),
    ],
)
def test_supported_arithmetic(expression: str, expected: str) -> None:
    assert calculator(expression) == expected


@pytest.mark.parametrize(
    "attack",
    [
        "().__class__.__mro__[1].__subclasses__()",
        "__import__('os').system('echo hacked')",
        "open('/etc/passwd').read()",
        "[x for x in range(10)]",
        "lambda: 1",
        "a + 1",
        "'abc' * 3",
        "True + 1",
        "1 if 1 else 2",
        "2 < 3",
    ],
)
def test_non_arithmetic_is_rejected(attack: str) -> None:
    with pytest.raises(ToolExecutionError, match="Invalid calculator expression"):
        calculator(attack)


@pytest.mark.parametrize(
    "expression", ["9**9**9**9", "2**101", "(10**50)**90", "99999999999999999999**100"]
)
def test_huge_results_are_rejected_without_computing(expression: str) -> None:
    with pytest.raises(ToolExecutionError):
        calculator(expression)


def test_division_by_zero() -> None:
    with pytest.raises(ToolExecutionError, match="Division by zero"):
        calculator("1/0")


def test_float_overflow() -> None:
    with pytest.raises(ToolExecutionError, match="out of range"):
        calculator("1e308 ** 2")


def test_non_real_result() -> None:
    with pytest.raises(ToolExecutionError, match="not a real number"):
        calculator("(-8) ** 0.5")


def test_length_limit() -> None:
    with pytest.raises(ToolExecutionError):
        calculator("1+" * MAX_EXPRESSION_LENGTH + "1")


def test_tool_error_is_still_a_value_error() -> None:
    """Backward compatibility: callers that caught ValueError keep working."""
    with pytest.raises(ValueError):
        calculator("")
