import asyncio

from aegis_platform.agent.graph import app_graph, tool_registry
from aegis_platform.tools.calculator import calculator
from scripts.stub_llm_provider import StubLLMProvider


async def main() -> None:
    tool_registry.register("calculator", calculator)

    cases = [
        {"user_input": "Hello Aegis! Explain what distributed systems are", "tool_used": "llm"},
        {"user_input": "calculate 2 + 2", "tool_used": "calculator", "response": "4"},
        {"user_input": "calculate 5 * 3", "tool_used": "calculator", "response": "15"},
        {"user_input": "calculate 10 / 2", "tool_used": "calculator", "response": "5.0"},
        {"user_input": "calculate 7 - 3", "tool_used": "calculator", "response": "4"},
    ]

    print("=== Agent Smoke Test ===\n")

    for case in cases:
        result = await app_graph.ainvoke(
            {"user_input": case["user_input"], "response": "", "tool_used": ""},
            config={"configurable": {"llm_provider": StubLLMProvider()}},
        )
        assert result["tool_used"] == case["tool_used"], result
        if "response" in case:
            assert result["response"] == case["response"], result

        print("Agent result:")
        print("User Input:", result["user_input"])
        print("Agent Output:", result["response"])
        print("Tools used:", result["tool_used"])
        print("=== End of Case ===")
        print()


if __name__ == "__main__":
    asyncio.run(main())
