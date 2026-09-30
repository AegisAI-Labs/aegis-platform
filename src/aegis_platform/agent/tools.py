def calculator(expression: str):
    """
    Temporary implementation for Day 3.

    RFC-005 will replace eval() with a safe parser.
    """
    try:
        return eval(expression)
    except (ArithmeticError, NameError, SyntaxError, TypeError, ValueError) as error:
        return str(error)
