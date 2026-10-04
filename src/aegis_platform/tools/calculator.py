def calculator(expression: str) -> str:
    """
    Evaluate a basic arithmetic expression.

    Day 4 implementation intentionally keeps the calculator
    simple. A safer expression parser can be introduced later.
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as exc:
        raise ValueError(f"Invalid calculator expression: {expression}") from exc

    return str(result)
