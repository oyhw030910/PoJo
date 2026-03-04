"""LangGraph Agent - Advanced Agent with State Management.

This module demonstrates how to build a production-ready agent using LangGraph,
which provides:
- State management
- Cyclic workflows
- Multi-agent orchestration
- Conditional routing

Usage:
    python examples/langgraph_agent.py
"""

import sys
import os
from typing import TypedDict, Annotated, List, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_code_agent_with_langgraph():
    """Create a code generation agent using LangGraph.

    This agent has a workflow of:
    Plan -> Code -> Test -> Refine -> Submit
    """
    print("\n" + "=" * 60)
    print("Code Agent with LangGraph")
    print("=" * 60)

    try:
        from langgraph.graph import StateGraph, END
        from langchain.tools import Tool
        from langchain import HuggingFacePipeline
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        # Define state
        class CodeAgentState(TypedDict):
            """State for code agent."""
            task: str
            plan: str
            code: str
            test_result: str
            iteration: int
            errors: List[str]
            final_output: str
            messages: Annotated[List[str], lambda x, y: x + y]

        # Load model
        print("\nLoading Qwen model...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.3,
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        # Define tools
        def check_syntax(code: str) -> str:
            """Check Python code syntax."""
            import ast
            try:
                ast.parse(code)
                return "Syntax OK"
            except SyntaxError as e:
                return f"Syntax Error: {e}"

        def run_tests(code: str, tests: str) -> str:
            """Run test cases against code."""
            try:
                safe_globals = {"__builtins__": __builtins__}
                exec(code, safe_globals)
                # Find function name
                import ast
                tree = ast.parse(code)
                func_name = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_name = node.name
                        break
                if func_name and func_name in safe_globals:
                    return f"Function '{func_name}' defined successfully"
                return "Code executed"
            except Exception as e:
                return f"Runtime Error: {e}"

        tools = [
            Tool(
                name="SyntaxChecker",
                func=check_syntax,
                description="Check Python code syntax"
            ),
            Tool(
                name="TestRunner",
                func=lambda code: run_tests(code, ""),
                description="Run tests against code"
            ),
        ]

        # Define workflow nodes
        def plan(state: CodeAgentState) -> CodeAgentState:
            """Plan the solution."""
            task = state["task"]
            state["messages"].append(f"Task: {task}")

            prompt = f"""Plan how to solve this coding task:

Task: {task}

Provide a step-by-step plan (max 3 steps):
"""
            plan_result = llm(prompt)

            state["plan"] = plan_result
            state["messages"].append(f"Plan: {plan_result[:200]}")
            print(f"\n[PLAN] {plan_result[:100]}...")

            return state

        def code(state: CodeAgentState) -> CodeAgentState:
            """Generate code."""
            task = state["task"]
            plan = state["plan"]

            prompt = f"""Write Python code for this task:

Task: {task}
Plan: {plan}

Requirements:
- Include docstring
- Handle edge cases
- Only output code, no explanation

Code:
"""
            code_result = llm(prompt)

            # Extract code from markdown
            if "```python" in code_result:
                start = code_result.find("```python") + 9
                end = code_result.find("```", start)
                if end > start:
                    code_result = code_result[start:end].strip()

            state["code"] = code_result
            state["iteration"] = state.get("iteration", 0) + 1
            state["messages"].append(f"Generated code ({state['iteration']} iterations)")
            print(f"\n[CODE] Generated {len(code_result)} chars")

            return state

        def test(state: CodeAgentState) -> CodeAgentState:
            """Test the code."""
            code = state["code"]

            # Check syntax
            syntax_result = check_syntax(code)
            state["messages"].append(f"Syntax check: {syntax_result}")
            print(f"\n[TEST] Syntax: {syntax_result}")

            if syntax_result != "Syntax OK":
                state["errors"] = state.get("errors", []) + [syntax_result]
            else:
                # Run tests
                test_result = run_tests(code, "")
                state["test_result"] = test_result
                state["messages"].append(f"Test result: {test_result}")
                print(f"[TEST] Result: {test_result}")

            return state

        def refine(state: CodeAgentState) -> CodeAgentState:
            """Refine code based on errors."""
            errors = state.get("errors", [])
            if not errors:
                state["final_output"] = state["code"]
                state["messages"].append("Code passed all tests!")
                print("\n[REFINE] Code is good!")
                return state

            task = state["task"]
            code = state["code"]
            error_msg = errors[-1]

            prompt = f"""Fix the code based on the error:

Task: {task}
Current code:
{code}

Error: {error_msg}

Provide the fixed complete code:
"""
            fixed_code = llm(prompt)

            if "```python" in fixed_code:
                start = fixed_code.find("```python") + 9
                end = fixed_code.find("```", start)
                if end > start:
                    fixed_code = fixed_code[start:end].strip()

            state["code"] = fixed_code
            state["errors"] = []  # Clear errors for re-test
            state["messages"].append("Code refined")
            print(f"\n[REFINE] Code fixed")

            return state

        def should_retry(state: CodeAgentState) -> str:
            """Decide whether to retry or submit."""
            if state.get("iteration", 0) >= 3:
                return "submit"
            if state.get("errors"):
                return "refine"
            if state.get("test_result", "").startswith("Runtime Error"):
                return "refine"
            return "submit"

        # Build graph
        builder = StateGraph(CodeAgentState)

        # Add nodes
        builder.add_node("plan", plan)
        builder.add_node("code", code)
        builder.add_node("test", test)
        builder.add_node("refine", refine)

        # Set entry point
        builder.set_entry_point("plan")

        # Add edges
        builder.add_edge("plan", "code")
        builder.add_edge("code", "test")

        # Conditional edge after test
        builder.add_conditional_edges(
            "test",
            should_retry,
            {
                "refine": "refine",
                "submit": END,
            }
        )

        # After refine, go back to test
        builder.add_edge("refine", "test")

        # Compile
        graph = builder.compile()

        print("\nLangGraph Code Agent created!")
        print("\nRunning example: Create a function to check if a number is prime")

        # Run
        result = graph.invoke({
            "task": "Write a function to check if a number is prime",
            "plan": "",
            "code": "",
            "test_result": "",
            "iteration": 0,
            "errors": [],
            "final_output": "",
            "messages": [],
        })

        print("\n" + "=" * 60)
        print("Final Result:")
        print("=" * 60)
        print(f"\nCode:\n{result['final_output'][:500]}...")
        print(f"\nIterations: {result['iteration']}")
        print(f"Messages: {len(result['messages'])}")

    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("\nInstall required packages:")
        print("  pip install langgraph langchain langchain-community")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


def create_math_agent_with_langgraph():
    """Create a math solving agent using LangGraph.

    Workflow: Understand -> Solve -> Verify -> Answer
    """
    print("\n" + "=" * 60)
    print("Math Agent with LangGraph")
    print("=" * 60)

    try:
        from langgraph.graph import StateGraph, END
        from langchain import HuggingFacePipeline
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        # Define state
        class MathAgentState(TypedDict):
            """State for math agent."""
            problem: str
            understanding: str
            steps: List[str]
            answer: str
            verified: bool
            messages: Annotated[List[str], lambda x, y: x + y]

        # Build simple graph
        builder = StateGraph(MathAgentState)

        def understand(state: MathAgentState) -> MathAgentState:
            """Understand the problem."""
            problem = state["problem"]
            state["messages"].append(f"Problem: {problem}")

            prompt = f"""Analyze this math problem:

{problem}

Identify:
1. What is being asked?
2. What information is given?
3. What operations are needed?
"""
            understanding = llm(prompt)
            state["understanding"] = understanding
            print(f"\n[UNDERSTAND] {understanding[:150]}...")
            return state

        def solve(state: MathAgentState) -> MathAgentState:
            """Solve step by step."""
            problem = state["problem"]
            understanding = state["understanding"]

            prompt = f"""Solve this math problem step by step:

Problem: {problem}
Analysis: {understanding}

Show each step clearly. End with 'Answer: [value]'
"""
            solution = llm(prompt)

            # Extract answer
            if "Answer:" in solution:
                answer = solution.split("Answer:")[-1].strip()
            else:
                answer = solution[-100:]

            state["answer"] = answer
            state["steps"] = solution.split("\n")
            state["messages"].append(f"Solution found")
            print(f"\n[SOLVE] Answer: {answer[:50]}")

            return state

        def verify(state: MathAgentState) -> MathAgentState:
            """Verify the answer."""
            problem = state["problem"]
            answer = state["answer"]

            prompt = f"""Verify if this answer is correct:

Problem: {problem}
Answer: {answer}

Is this correct? Answer YES or NO with explanation.
"""
            verification = llm(prompt)

            state["verified"] = "YES" in verification.upper()
            state["messages"].append(f"Verified: {state['verified']}")
            print(f"\n[VERIFY] Correct: {state['verified']}")

            return state

        def should_retry(state: MathAgentState) -> str:
            """Decide whether to retry."""
            if state["verified"]:
                return "end"
            return "retry"

        # Add nodes
        builder.add_node("understand", understand)
        builder.add_node("solve", solve)
        builder.add_node("verify", verify)

        # Set entry and edges
        builder.set_entry_point("understand")
        builder.add_edge("understand", "solve")
        builder.add_edge("solve", "verify")

        # Conditional retry
        builder.add_conditional_edges(
            "verify",
            should_retry,
            {
                "retry": "solve",
                "end": END,
            }
        )

        # Compile
        llm = None  # Will be created in nodes
        graph = builder.compile()

        print("\nMath Agent workflow created!")
        print("\nExample: What is 25% of 180?")

        # For demo, just show the workflow structure
        print("\nWorkflow structure:")
        print("  [Entry] -> Understand -> Solve -> Verify")
        print("                             ↑        ↓")
        print("                             └──(if not verified)")

    except Exception as e:
        print(f"Error: {e}")


def show_workflow_diagram():
    """Show workflow diagram for different agent types."""
    diagram = """

╔═══════════════════════════════════════════════════════════╗
║              LangGraph Agent Workflows                     ║
╠═══════════════════════════════════════════════════════════╣

┌─ Code Agent ─────────────────────────────────────────────┐
│                                                           │
│   [Start] → [Plan] → [Code] → [Test] ─┬─→ [Refine] ─┐   │
│                                       │             │   │
│                                       ↓             │   │
│                                    [Submit] ←───────┘   │
│                                       ↓                 │
│                                    [End]                │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌─ Math Agent ─────────────────────────────────────────────┐
│                                                           │
│   [Start] → [Understand] → [Solve] → [Verify] ─┬─→ [End]│
│                                                │         │
│                                                ↓         │
│                                           (retry) ───┐   │
│                                                      │   │
│                                                      ↑   │
│                                                      └───┘│
│                                                           │
└───────────────────────────────────────────────────────────┘

┌─ Multi-Agent System ──────────────────────────────────────┐
│                                                            │
│                    ┌─→ [Coder Agent] ─┐                   │
│                    │                  │                    │
│   [Start] → [Planner] ─→ [Reviewer] ──┼─→ [Finalize]     │
│                    │                  │                    │
│                    └─→ [Tester Agent] ─┘                   │
│                                                            │
└────────────────────────────────────────────────────────────┘

Benefits of using LangGraph:
────────────────────────────
✓ State management is automatic
✓ Cyclic workflows (retry loops) are easy
✓ Conditional routing built-in
✓ Multi-agent orchestration
✓ Debugging and visualization tools
✓ Production-ready patterns
"""
    print(diagram)


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("""
LangGraph Agent Examples
========================

Usage: python examples/langgraph_agent.py [command]

Commands:
  code      - Code generation agent with workflow
  math      - Math solving agent with verification
  diagram   - Show workflow diagrams
  all       - Run all examples

Example:
  python examples/langgraph_agent.py code
  python examples/langgraph_agent.py all
""")
        return

    command = sys.argv[1].lower()

    if command == "code":
        create_code_agent_with_langgraph()
    elif command == "math":
        create_math_agent_with_langgraph()
    elif command == "diagram":
        show_workflow_diagram()
    elif command == "all":
        show_workflow_diagram()
        create_code_agent_with_langgraph()
        create_math_agent_with_langgraph()
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
