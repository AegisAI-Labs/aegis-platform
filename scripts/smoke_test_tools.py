from services.agent.tools import calculator

cases = [
    "2+2",
    "5*3",
    "10/2",
    "7-3",
]

for expression in cases:
    result = calculator(expression)
    print(f"Expression: {expression} => Result: {result}")
    print("=== End of Case ===")
    print()
