from aegis_platform.agent.graph import app_graph

cases = [
    {"user_input": "Hello Aegis!"},
    {"user_input": "calc: 2 + 2"},
    {"user_input": "calc: 5 * 3"},
    {"user_input": "calc: 10 / 2"},
    {"user_input": "calc: 7 - 3"},
]

print("=== Agent Smoke Test ===\n")

for case in cases:
    result = app_graph.invoke(case)
    print("Agent result:")
    print("User Input:", result["user_input"])
    print("Agent Output:", result["response"])
    print("Tools used:", result["tool_used"])
    print("=== End of Case ===")
    print()

# To run the smoke test for the agent:
# uv run python -m scripts.smoke_test_agent
