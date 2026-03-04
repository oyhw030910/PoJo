"""LangChain Tools Integration Module.

Provides integration with LangChain tools for LLM agents.
"""

import json
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class LangChainToolWrapper:
    """Wrapper for a LangChain tool.

    Attributes:
        name: Tool name
        description: Tool description
        langchain_tool: The original LangChain tool
        parameters: Parameter schema
    """
    name: str
    description: str
    langchain_tool: Any
    parameters: Dict[str, Any] = field(default_factory=dict)

    def execute(self, **kwargs) -> Any:
        """Execute the LangChain tool.

        Args:
            **kwargs: Tool arguments

        Returns:
            Tool output
        """
        return self.langchain_tool.run(kwargs)


class LangChainTools:
    """Manager for LangChain tool integration.

    Provides seamless integration with LangChain's extensive tool library.
    """

    def __init__(self):
        """Initialize LangChain tools integration."""
        self._tools: Dict[str, LangChainToolWrapper] = {}
        self._available_tools: List[str] = []
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize LangChain integration."""
        if self._initialized:
            return

        try:
            from langchain.agents import load_tools, Tool
            from langchain.tools import BaseTool

            self._langchain_available = True
            self._Tool = Tool
            self._BaseTool = BaseTool
            self._load_tools = load_tools

        except ImportError:
            self._langchain_available = False
            print("LangChain not installed. LangChain tools will not be available.")

        self._initialized = True

    def is_available(self) -> bool:
        """Check if LangChain is available.

        Returns:
            True if LangChain is installed
        """
        self._initialize()
        return self._langchain_available

    def load_tools(
        self,
        tool_names: List[str],
        llm: Optional[Any] = None,
        **kwargs
    ) -> List[str]:
        """Load LangChain tools by name.

        Args:
            tool_names: List of tool names to load
            llm: Optional LLM for tools that require it
            **kwargs: Additional arguments

        Returns:
            List of loaded tool names
        """
        self._initialize()

        if not self._langchain_available:
            return []

        try:
            tools = self._load_tools(tool_names, llm=llm, **kwargs)

            loaded = []
            for tool in tools:
                name = tool.name if hasattr(tool, 'name') else tool.__class__.__name__
                description = tool.description if hasattr(tool, 'description') else ""

                wrapper = LangChainToolWrapper(
                    name=name,
                    description=description,
                    langchain_tool=tool,
                )

                self._tools[name] = wrapper
                self._available_tools.append(name)
                loaded.append(name)

            return loaded

        except Exception as e:
            print(f"Error loading LangChain tools: {e}")
            return []

    def register_tool(
        self,
        tool: Any,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """Register a custom LangChain tool.

        Args:
            tool: LangChain tool instance
            name: Optional name override
            description: Optional description override

        Returns:
            Registered tool name
        """
        self._initialize()

        tool_name = name or (tool.name if hasattr(tool, 'name') else tool.__class__.__name__)
        tool_desc = description or (tool.description if hasattr(tool, 'description') else "")

        wrapper = LangChainToolWrapper(
            name=tool_name,
            description=tool_desc,
            langchain_tool=tool,
        )

        self._tools[tool_name] = wrapper
        self._available_tools.append(tool_name)

        return tool_name

    def get_tool(self, name: str) -> Optional[LangChainToolWrapper]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool wrapper or None
        """
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tools.

        Returns:
            List of tool names
        """
        return self._available_tools

    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool arguments

        Returns:
            Dictionary with result
        """
        tool = self.get_tool(tool_name)

        if tool is None:
            return {
                "success": False,
                "error": f"Tool not found: {tool_name}",
            }

        try:
            result = tool.execute(**kwargs)

            return {
                "success": True,
                "output": result,
                "tool": tool_name,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
            }

    def get_tool_prompt(self) -> str:
        """Generate a prompt describing available tools.

        Returns:
            Formatted prompt string
        """
        if not self._tools:
            return "No tools available."

        lines = ["Available LangChain tools:"]

        for name, tool in self._tools.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Description: {tool.description}")

        return "\n".join(lines)

    def create_custom_tool(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Optional[Dict[str, Any]] = None
    ) -> LangChainToolWrapper:
        """Create a custom tool from a function.

        Args:
            name: Tool name
            description: Tool description
            func: Function to wrap
            parameters: Optional parameter schema

        Returns:
            Created tool wrapper
        """
        self._initialize()

        if self._langchain_available:
            # Create LangChain tool
            tool = self._Tool(
                name=name,
                description=description,
                func=func,
            )
        else:
            # Create simple wrapper
            class SimpleTool:
                def __init__(self, name, description, func):
                    self.name = name
                    self.description = description
                    self._func = func

                def run(self, input_data):
                    return self._func(**input_data)

            tool = SimpleTool(name, description, func)

        wrapper = LangChainToolWrapper(
            name=name,
            description=description,
            langchain_tool=tool,
            parameters=parameters or {},
        )

        self._tools[name] = wrapper
        if name not in self._available_tools:
            self._available_tools.append(name)

        return wrapper


# Common LangChain tool sets
COMMON_TOOL_SETS = {
    "all": [
        "python_repl_ast",
        "llm-math",
        "wikipedia",
        "requests",
        "terminal",
    ],
    "calculation": [
        "llm-math",
        "python_repl_ast",
    ],
    "research": [
        "wikipedia",
        "requests",
        "search",
    ],
    "code": [
        "python_repl_ast",
        "terminal",
    ],
}


def get_common_tools(
    tool_set: str = "calculation",
    llm: Optional[Any] = None
) -> LangChainTools:
    """Get a common set of tools.

    Args:
        tool_set: Name of tool set
        llm: Optional LLM for tools

    Returns:
        LangChainTools instance with loaded tools
    """
    lc_tools = LangChainTools()

    if tool_set not in COMMON_TOOL_SETS:
        tool_set = "calculation"

    tool_names = COMMON_TOOL_SETS[tool_set]
    lc_tools.load_tools(tool_names, llm=llm)

    return lc_tools


# Create default instance
_default_lc_tools = LangChainTools()


def get_default_tools() -> LangChainTools:
    """Get the default LangChain tools instance.

    Returns:
        Default instance
    """
    return _default_lc_tools


# Convenience functions for common tools
def search_wikipedia(query: str, top_k: int = 1) -> Dict[str, Any]:
    """Search Wikipedia.

    Args:
        query: Search query
        top_k: Number of results

    Returns:
        Search results
    """
    try:
        from langchain.tools import WikipediaQueryRun
        from langchain.utilities import WikipediaAPIWrapper

        api = WikipediaAPIWrapper(top_k_results=top_k)
        tool = WikipediaQueryRun(api_wrapper=api)

        result = tool.run(query)

        return {
            "success": True,
            "result": result,
        }
    except ImportError:
        return {
            "success": False,
            "error": "Wikipedia tool not available",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def run_python(code: str) -> Dict[str, Any]:
    """Run Python code.

    Args:
        code: Python code to execute

    Returns:
        Execution result
    """
    try:
        from langchain.tools import PythonREPLTool

        tool = PythonREPLTool()
        result = tool.run(code)

        return {
            "success": True,
            "result": result,
        }
    except ImportError:
        return {
            "success": False,
            "error": "Python REPL tool not available",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
