"""Quick Start Script for RL-LLM Agent.

This script provides a minimal example of using the Agent framework
without requiring all dependencies.

Usage:
    python examples/quick_start.py [code|math|chat|memory]
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quick_start_code():
    """Quick start with Code Agent."""
    print("\n" + "=" * 50)
    print("Code Agent Quick Start")
    print("=" * 50)

    from examples.code_agent import CodeAgent

    agent = CodeAgent()

    print("\nTask: Generate a function to check if a number is even")
    result = agent.generate_code(
        description="Write a function that checks if a number is even",
        test_cases=[
            {"args": [2], "expected": True},
            {"args": [3], "expected": False},
        ],
        verbose=False,
    )

    print(f"\nResult: Success = {result.get('success', False)}")
    print(f"Code:\n{result.get('code', 'N/A')}")


def quick_start_math():
    """Quick start with Math Agent."""
    print("\n" + "=" * 50)
    print("Math Agent Quick Start")
    print("=" * 50)

    from examples.math_agent import MathAgent

    agent = MathAgent()

    problems = [
        ("What is 25 + 17?", "arithmetic"),
        ("What is 20% of 50?", "percentage"),
    ]

    for problem, topic in problems:
        print(f"\nProblem: {problem}")
        result = agent.solve(problem, topic=topic, verbose=False)
        print(f"Answer: {result.get('answer', 'N/A')}")


def quick_start_chat():
    """Quick start with Chat Agent."""
    print("\n" + "=" * 50)
    print("Chat Agent Quick Start")
    print("=" * 50)

    from agent.llm_wrapper import LLMWrapper, LLMConfig

    llm = LLMWrapper(LLMConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct"))

    print("\nAsking: 'What is machine learning?'")
    response = llm.generate("Explain what machine learning is in simple terms.", max_new_tokens=100)
    print(f"\nResponse: {response[:200]}...")


def quick_start_memory():
    """Quick start with Memory."""
    print("\n" + "=" * 50)
    print("Memory Quick Start")
    print("=" * 50)

    from agent.memory import MemoryManager

    memory = MemoryManager()

    print("\nAdding experiences...")
    memory.add_experience("User asked about Python", "Explained basics", 0.8)
    memory.add_experience("User asked about RL", "Explained concepts", 0.9)

    print("Retrieving context...")
    context = memory.get_context(num_recent=1)
    print(f"Context: {context[:100] if context else 'None'}...")

    stats = memory.get_stats()
    print(f"\nStats: {stats}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python quick_start.py [code|math|chat|memory|all]")
        print("\nRunning all examples...")
        examples = ["code", "math", "chat", "memory"]
    else:
        examples = [sys.argv[1]]

    actions = {
        "code": quick_start_code,
        "math": quick_start_math,
        "chat": quick_start_chat,
        "memory": quick_start_memory,
    }

    for ex in examples:
        if ex in actions:
            try:
                actions[ex]()
            except Exception as e:
                print(f"Error running {ex}: {e}")
        else:
            print(f"Unknown example: {ex}")

    print("\n" + "=" * 50)
    print("Quick Start Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
