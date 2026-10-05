from aegis_platform.tools.calculator import calculator

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

# To run the smoke test for the tools:
# uv run python -m scripts.smoke_test_tools
