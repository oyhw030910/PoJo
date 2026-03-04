"""LangChain Agent Example.

This module demonstrates how to create agents using LangChain and LangGraph,
which significantly simplifies agent development.

Usage:
    python examples/langchain_agent.py [basic|react|graph]
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_basic_langchain_agent():
    """Create a basic agent using LangChain.

    This shows how LangChain simplifies agent creation compared to
    building everything from scratch.
    """
    print("\n" + "=" * 60)
    print("Basic LangChain Agent")
    print("=" * 60)

    try:
        from langchain.agents import AgentType, initialize_agent, load_tools
        from langchain.llms import HuggingFaceHub
        from langchain import HuggingFacePipeline
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        print("\nLoading Qwen model...")

        # Create HuggingFace pipeline
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
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
        )

        # Wrap in LangChain
        llm = HuggingFacePipeline(pipeline=pipe)

        # Load tools - LangChain provides many built-in tools
        tools = load_tools(["llm-math"], llm=llm)

        # Create agent - just one line!
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

        print("\nAgent created successfully!")
        print("\nRunning example: Calculate 25% of 180")
        result = agent.run("What is 25% of 180?")
        print(f"Result: {result}")

    except ImportError as e:
        print(f"LangChain not fully installed: {e}")
        print("\nInstall with: pip install langchain langchain-community")
    except Exception as e:
        print(f"Error: {e}")


def create_react_agent():
    """Create a ReAct agent using LangChain.

    ReAct (Reasoning + Acting) is a powerful agent pattern that
    LangChain makes easy to implement.
    """
    print("\n" + "=" * 60)
    print("ReAct Agent with LangChain")
    print("=" * 60)

    try:
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain import HuggingFacePipeline
        from langchain.tools import Tool
        from langchain.prompts import PromptTemplate
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        print("\nLoading model and creating tools...")

        # Load model
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

        # Define custom tools - very simple!
        def calculator(expr: str) -> str:
            """Calculate mathematical expression."""
            try:
                # Safe evaluation
                result = eval(expr, {"__builtins__": {}}, {"abs": abs, "round": round})
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {e}"

        def code_executor(code: str) -> str:
            """Execute Python code."""
            try:
                safe_globals = {"__builtins__": __builtins__}
                exec(code, safe_globals)
                return "Code executed successfully"
            except Exception as e:
                return f"Error: {e}"

        tools = [
            Tool(
                name="Calculator",
                func=calculator,
                description="Useful for mathematical calculations. Input: expression like '2+2' or '0.25*180'"
            ),
            Tool(
                name="CodeExecutor",
                func=code_executor,
                description="Useful for executing Python code. Input: valid Python code"
            ),
        ]

        # ReAct prompt template
        prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")

        # Create ReAct agent
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
        )

        print("\nReAct Agent created!")
        print("\nRunning example: Calculate 15% of 200")
        result = agent_executor.invoke({"input": "What is 15% of 200?"})
        print(f"\nFinal Answer: {result.get('output', 'N/A')}")

    except Exception as e:
        print(f"Error: {e}")


def create_langgraph_agent():
    """Create an agent using LangGraph.

    LangGraph provides a more structured way to build complex agent workflows
    with state management and cyclic graphs.
    """
    print("\n" + "=" * 60)
    print("LangGraph Agent")
    print("=" * 60)

    try:
        # Check if langgraph is installed
        from langgraph.graph import StateGraph, END
        from typing import TypedDict, Annotated
        import operator

        print("\nLangGraph is available!")

        # Define state schema
        class AgentState(TypedDict):
            """State for the agent."""
            input: str
            thought: str
            action: str
            action_input: str
            observation: str
            output: str
            messages: Annotated[list, operator.add]

        # Create graph builder
        builder = StateGraph(AgentState)

        # Define nodes
        def think(state: AgentState) -> AgentState:
            """Think about what to do."""
            print(f"\n[THINK] Input: {state['input']}")
            state["thought"] = "Analyzing the input..."
            state["messages"].append(f"User: {state['input']}")
            return state

        def act(state: AgentState) -> AgentState:
            """Take an action."""
            print(f"[ACT] Thought: {state['thought']}")
            state["action"] = "calculate"
            state["action_input"] = "extract from input"
            return state

        def observe(state: AgentState) -> AgentState:
            """Observe the result."""
            print(f"[OBSERVE] Action: {state['action']}")
            state["observation"] = "Result computed"
            return state

        def respond(state: AgentState) -> AgentState:
            """Generate final response."""
            print(f"[RESPOND] Observation: {state['observation']}")
            state["output"] = f"Based on my analysis: {state['input']}"
            state["messages"].append(f"Assistant: {state['output']}")
            return state

        # Add nodes to graph
        builder.add_node("think", think)
        builder.add_node("act", act)
        builder.add_node("observe", observe)
        builder.add_node("respond", respond)

        # Set entry point
        builder.set_entry_point("think")

        # Add edges (workflow)
        builder.add_edge("think", "act")
        builder.add_edge("act", "observe")
        builder.add_edge("observe", "respond")
        builder.add_edge("respond", END)

        # Compile graph
        graph = builder.compile()

        print("\nLangGraph Agent created!")
        print("\nRunning workflow...")

        # Run the agent
        result = graph.invoke({
            "input": "What is 10% of 500?",
            "messages": [],
            "thought": "",
            "action": "",
            "action_input": "",
            "observation": "",
            "output": "",
        })

        print(f"\nFinal Output: {result['output']}")
        print(f"Messages: {result['messages']}")

    except ImportError:
        print("\nLangGraph not installed.")
        print("Install with: pip install langgraph")
        print("\nLangGraph enables:")
        print("- State management")
        print("- Cyclic workflows")
        print("- Multi-agent systems")
        print("- Conditional routing")
    except Exception as e:
        print(f"Error: {e}")


def compare_approaches():
    """Compare different approaches to building agents."""
    print("\n" + "=" * 60)
    print("Agent Building Approaches Comparison")
    print("=" * 60)

    comparison = """
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Building Approaches                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. From Scratch (Original Code)                                │
│     ├── Full control                                            │
│     ├── More code to write                                      │
│     ├── Need to handle edge cases                               │
│     └── Best for learning and customization                     │
│                                                                  │
│  2. LangChain                                                   │
│     ├── Less code (10x reduction)                               │
│     ├── Built-in tools (100+ available)                         │
│     ├── Multiple agent types (ReAct, MRKL, etc.)                │
│     └── Best for rapid prototyping                              │
│                                                                  │
│  3. LangGraph                                                   │
│     ├── State management built-in                               │
│     ├── Cyclic workflows support                                │
│     ├── Multi-agent orchestration                               │
│     └── Best for complex workflows                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Code Comparison:
─────────────────────────────────────────────────────────────────

From Scratch (~100+ lines):
    class CodeAgent:
        def __init__(self):
            self.llm = LLMWrapper(...)
            self.memory = MemoryManager()
            self.tools = ToolManager()
            # ... lots of setup

        def generate_code(self, description):
            # Manual prompt building
            # Manual tool calling
            # Manual error handling
            pass

LangChain (~10 lines):
    from langchain.agents import initialize_agent, load_tools

    tools = load_tools(["python_repl"], llm=llm)
    agent = initialize_agent(tools, llm, agent="zero-shot-react")
    result = agent.run("Write code to calculate factorial")

LangGraph (~20 lines for complex workflow):
    from langgraph.graph import StateGraph, END

    builder = StateGraph(AgentState)
    builder.add_node("think", think_func)
    builder.add_node("act", act_func)
    builder.add_edge("think", "act")
    graph = builder.compile()
    result = graph.invoke({"input": task})
"""

    print(comparison)


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("""
LangChain Agent Examples
========================

Usage: python examples/langchain_agent.py [command]

Commands:
  basic     - Basic LangChain agent with tools
  react     - ReAct agent with custom tools
  graph     - LangGraph agent with state management
  compare   - Compare different approaches
  all       - Run all examples

Example:
  python examples/langchain_agent.py basic
  python examples/langchain_agent.py all
""")
        return

    command = sys.argv[1].lower()

    examples = {
        "basic": create_basic_langchain_agent,
        "react": create_react_agent,
        "graph": create_langgraph_agent,
        "compare": compare_approaches,
    }

    if command == "all":
        for name, func in examples.items():
            try:
                func()
            except Exception as e:
                print(f"Error in {name}: {e}")
    elif command in examples:
        examples[command]()
    else:
        print(f"Unknown command: {command}")
        print("Use 'all', 'basic', 'react', 'graph', or 'compare'")


if __name__ == "__main__":
    main()
