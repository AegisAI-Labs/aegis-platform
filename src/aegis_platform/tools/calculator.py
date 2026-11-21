"""Safe arithmetic calculator tool.

Expressions are parsed with ``ast`` and evaluated against a whitelist of numeric nodes and
operators. Names, calls, attributes, subscripts and every other construct are rejected, so the
tool cannot execute arbitrary Python (it previously used ``eval``). Size limits stop inputs such as
``9**9**9**9`` from exhausting CPU or memory.
"""

import ast
import operator
from collections.abc import Callable

from .errors import ToolExecutionError

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 100
MAX_RESULT_BITS = 4096

Number = int | float

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _invalid(expression: str) -> ToolExecutionError:
    return ToolExecutionError(f"Invalid calculator expression: {expression}")


def _check_size(value: Number) -> Number:
    if isinstance(value, int) and value.bit_length() > MAX_RESULT_BITS:
        raise ToolExecutionError("Calculator result is too large")
    return value


def _power(base: Number, exponent: Number) -> Number:
    if abs(exponent) > MAX_EXPONENT:
        raise ToolExecutionError(
            f"Calculator exponent must be between -{MAX_EXPONENT} and {MAX_EXPONENT}"
        )
    # Reject before computing: the result would have about bit_length * exponent bits.
    if (
        isinstance(base, int)
        and isinstance(exponent, int)
        and exponent > 0
        and base.bit_length() * exponent > MAX_RESULT_BITS
    ):
        raise ToolExecutionError("Calculator result is too large")
    return base**exponent


def _evaluate(node: ast.AST, expression: str) -> Number:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        if isinstance(node.value, bool):  # True/False are ints in Python; not numbers here
            raise _invalid(expression)
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand, expression))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left, expression)
        right = _evaluate(node.right, expression)
        if isinstance(node.op, ast.Pow):
            return _check_size(_power(left, right))
        return _check_size(_BINARY_OPERATORS[type(node.op)](left, right))

    raise _invalid(expression)


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the result as text."""
    expression = expression.strip()

    if not expression or len(expression) > MAX_EXPRESSION_LENGTH:
        raise _invalid(expression[:MAX_EXPRESSION_LENGTH])

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise _invalid(expression) from exc

    try:
        result = _evaluate(tree.body, expression)
    except ZeroDivisionError as exc:
        raise ToolExecutionError("Division by zero") from exc
    except (OverflowError, ValueError) as exc:
        if isinstance(exc, ToolExecutionError):
            raise
        raise ToolExecutionError("Calculator result is out of range") from exc

    if isinstance(result, complex):  # e.g. (-8) ** 0.5
        raise ToolExecutionError("Calculator result is not a real number")

    return str(result)
