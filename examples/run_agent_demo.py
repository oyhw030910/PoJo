"""Agent Demo Script.

This script provides a quick demo of different Agent types available in the framework.

Usage:
    python examples/run_agent_demo.py [agent_type]

Agent types:
    - code: Code generation agent demo
    - math: Math solving agent demo
    - chat: Chat agent demo
    - all: Run all demos
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_code_agent_demo():
    """Run code agent demo."""
    print("\n" + "=" * 60)
    print("Code Agent Demo")
    print("=" * 60)

    try:
        from examples.code_agent import CodeAgent, CodeAgentConfig

        # Create agent
        print("\nCreating Code Agent...")
        config = CodeAgentConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            max_iterations=3,
            auto_test=True,
        )
        agent = CodeAgent(config)

        # Generate code
        print("\nGenerating code for: 'Write a function to check if a number is prime'")
        result = agent.generate_code(
            description="Write a function to check if a number is prime",
            test_cases=[
                {"args": [2], "expected": True},
                {"args": [17], "expected": True},
                {"args": [4], "expected": False},
                {"args": [1], "expected": False},
            ],
            verbose=True,
        )

        print(f"\nResult:")
        print(f"  Success: {result.get('success', False)}")
        print(f"  Iterations: {result.get('iterations', 1)}")
        print(f"\nGenerated Code:")
        print(f"```python\n{result.get('code', 'N/A')}\n```")

    except ImportError as e:
        print(f"Error: Could not import CodeAgent - {e}")
    except Exception as e:
        print(f"Error during demo: {e}")


def run_math_agent_demo():
    """Run math agent demo."""
    print("\n" + "=" * 60)
    print("Math Agent Demo")
    print("=" * 60)

    try:
        from examples.math_agent import MathAgent, MathAgentConfig

        # Create agent
        print("\nCreating Math Agent...")
        config = MathAgentConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            step_by_step=True,
            use_calculator=True,
        )
        agent = MathAgent(config)

        # Solve problems
        problems = [
            ("What is 25 + 17 + 38?", "arithmetic"),
            ("What is 15% of 80?", "percentage"),
            ("What is the area of a rectangle with length 8 and width 5?", "geometry"),
        ]

        for problem, topic in problems:
            print(f"\nProblem: {problem}")
            result = agent.solve(problem, topic=topic, verbose=True)
            print(f"Answer: {result.get('answer', 'N/A')}")
            print(f"Method: {result.get('method', 'N/A')}")

    except ImportError as e:
        print(f"Error: Could not import MathAgent - {e}")
    except Exception as e:
        print(f"Error during demo: {e}")


def run_chat_agent_demo():
    """Run chat agent demo."""
    print("\n" + "=" * 60)
    print("Chat Agent Demo")
    print("=" * 60)

    try:
        from agent.llm_wrapper import LLMWrapper, LLMConfig

        # Create LLM
        print("\nCreating LLM...")
        llm = LLMWrapper(LLMConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct"))

        # Chat loop
        print("\nChat Agent Ready! (type 'quit' to exit)")
        print("-" * 40)

        conversation_history = []

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Build prompt with history
            prompt = "You are a helpful AI assistant.\n\n"
            for msg in conversation_history[-5:]:  # Last 5 messages
                prompt += f"{msg['role']}: {msg['content']}\n"
            prompt += f"You: {user_input}\nAssistant: "

            # Generate response
            response = llm.generate(prompt, max_new_tokens=256)

            print(f"Agent: {response}")

            # Save to history
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})

    except ImportError as e:
        print(f"Error: Could not import LLMWrapper - {e}")
    except Exception as e:
        print(f"Error during demo: {e}")


def run_task_agent_demo():
    """Run task agent demo."""
    print("\n" + "=" * 60)
    print("Task Agent Demo")
    print("=" * 60)

    try:
        from agent.llm_wrapper import LLMWrapper, LLMConfig
        from agent.planner import Planner, PlannerConfig

        # Create components
        print("\nCreating Task Agent...")
        llm = LLMWrapper(LLMConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct"))
        planner = Planner(llm, config=PlannerConfig(method="react", max_iterations=5))

        # Example task
        task = "Plan a healthy breakfast menu"

        print(f"\nTask: {task}")
        print("\nGenerating plan...")

        result = planner.plan(task)

        if result.plan and result.plan.thoughts:
            print("\nPlan:")
            for i, thought in enumerate(result.plan.thoughts, 1):
                print(f"  {i}. {thought.content}")
        else:
            print("Could not generate plan.")

    except ImportError as e:
        print(f"Error: Could not import components - {e}")
    except Exception as e:
        print(f"Error during demo: {e}")


def run_memory_agent_demo():
    """Run memory agent demo."""
    print("\n" + "=" * 60)
    print("Memory Agent Demo")
    print("=" * 60)

    try:
        from agent.memory import MemoryManager, MemoryConfig

        # Create memory
        print("\nCreating Memory Manager...")
        config = MemoryConfig(
            short_term_size=10,
            long_term_enabled=True,
            long_term_top_k=3,
        )
        memory = MemoryManager(config)

        # Add experiences
        print("\nAdding experiences...")
        experiences = [
            ("User asked about Python", "Explained Python basics", 0.8),
            ("User asked about RL", "Explained reinforcement learning", 0.9),
            ("User asked about code", "Provided code example", 0.7),
        ]

        for obs, action, reward in experiences:
            memory.add_experience(observation=obs, action=action, reward=reward)
            print(f"  Added: {obs}")

        # Get context
        print("\nRetrieving context...")
        context = memory.get_context(num_recent=2)
        print(f"Context:\n{context}")

        # Get stats
        print("\nMemory Stats:")
        stats = memory.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    except ImportError as e:
        print(f"Error: Could not import MemoryManager - {e}")
    except Exception as e:
        print(f"Error during demo: {e}")


def main():
    """Main function."""
    agent_type = sys.argv[1] if len(sys.argv) > 1 else "all"

    demos = {
        "code": run_code_agent_demo,
        "math": run_math_agent_demo,
        "chat": run_chat_agent_demo,
        "task": run_task_agent_demo,
        "memory": run_memory_agent_demo,
    }

    if agent_type == "all":
        for demo_func in demos.values():
            demo_func()
    elif agent_type in demos:
        demos[agent_type]()
    else:
        print(f"Unknown agent type: {agent_type}")
        print(f"Available types: {', '.join(demos.keys())}, all")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
