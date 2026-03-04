"""Tool Manager Module.

Manages tool registration, discovery, and execution for LLM agents.
"""

import json
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from functools import wraps
import re


@dataclass
class ToolDefinition:
    """Definition of a tool.

    Attributes:
        name: Tool name
        description: Tool description
        function: The tool function
        parameters: Parameter schema
        returns: Return type description
        category: Tool category
        examples: Usage examples
    """
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]
    returns: str = ""
    category: str = "general"
    examples: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """Result of tool execution.

    Attributes:
        success: Whether execution was successful
        output: Tool output
        error: Error message if any
        metadata: Additional metadata
    """
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Registry for tool discovery and management.

    Provides centralized registration and lookup of tools.
    """

    def __init__(self):
        """Initialize tool registry."""
        self._tools: Dict[str, ToolDefinition] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: str = "general",
        examples: Optional[List[str]] = None,
    ) -> Callable:
        """Decorator to register a tool.

        Args:
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            category: Tool category
            examples: Usage examples

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or ""

            # Extract parameters from function signature
            sig = inspect.signature(func)
            parameters = {}
            for param_name, param in sig.parameters.items():
                param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
                default = param.default if param.default != inspect.Parameter.empty else None
                parameters[param_name] = {
                    "type": str(param_type),
                    "default": default,
                }

            # Get return type
            returns = ""
            if sig.return_annotation != inspect.Signature.empty:
                returns = str(sig.return_annotation)

            # Create tool definition
            tool_def = ToolDefinition(
                name=tool_name,
                description=tool_desc.strip(),
                function=func,
                parameters=parameters,
                returns=returns,
                category=category,
                examples=examples or [],
            )

            # Register
            self._tools[tool_name] = tool_def

            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(tool_name)

            return func

        return decorator

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool definition or None
        """
        return self._tools.get(name)

    def get_tools(self, names: List[str]) -> List[ToolDefinition]:
        """Get multiple tools by name.

        Args:
            names: Tool names

        Returns:
            List of tool definitions
        """
        return [self._tools[name] for name in names if name in self._tools]

    def get_tools_by_category(self, category: str) -> List[ToolDefinition]:
        """Get tools by category.

        Args:
            category: Category name

        Returns:
            List of tool definitions
        """
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def list_tools(self) -> List[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def list_categories(self) -> List[str]:
        """List all categories.

        Returns:
            List of category names
        """
        return list(self._categories.keys())

    def get_tool_prompt(self, tools: Optional[List[str]] = None) -> str:
        """Generate a prompt describing available tools.

        Args:
            tools: Optional list of tool names to include

        Returns:
            Formatted prompt string
        """
        if tools:
            tool_defs = self.get_tools(tools)
        else:
            tool_defs = list(self._tools.values())

        lines = ["Available tools:"]

        for tool in tool_defs:
            lines.append(f"\n{tool.name}:")
            lines.append(f"  Description: {tool.description}")
            if tool.parameters:
                lines.append("  Parameters:")
                for param_name, param_info in tool.parameters.items():
                    default = param_info.get("default", "required")
                    lines.append(f"    - {param_name}: {param_info['type']} (default: {default})")
            if tool.examples:
                lines.append("  Examples:")
                for example in tool.examples:
                    lines.append(f"    {example}")

        return "\n".join(lines)

    def format_tool_call(self, name: str, **kwargs) -> str:
        """Format a tool call as string.

        Args:
            name: Tool name
            **kwargs: Tool arguments

        Returns:
            Formatted tool call string
        """
        args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in kwargs.items())
        return f"{name}({args_str})"

    def parse_tool_call(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Parse a tool call from text.

        Args:
            text: Text containing tool call

        Returns:
            Tuple of (tool_name, arguments) or None
        """
        # Pattern: tool_name(arg1=value1, arg2=value2, ...)
        pattern = r"(\w+)\(([^)]*)\)"
        match = re.search(pattern, text)

        if not match:
            return None

        tool_name = match.group(1)
        args_str = match.group(2)

        # Parse arguments
        args = {}
        if args_str.strip():
            # Simple argument parsing (handles strings, numbers, booleans)
            arg_pattern = r'(\w+)=("[^"]*"|\'[^\']*\'|\d+\.?\d*|true|false|null)'
            for arg_match in re.finditer(arg_pattern, args_str):
                arg_name = arg_match.group(1)
                arg_value = arg_match.group(2)

                # Parse value
                if arg_value.startswith('"') or arg_value.startswith("'"):
                    arg_value = arg_value[1:-1]
                elif arg_value == "true":
                    arg_value = True
                elif arg_value == "false":
                    arg_value = False
                elif arg_value == "null":
                    arg_value = None
                else:
                    try:
                        arg_value = float(arg_value)
                        if arg_value == int(arg_value):
                            arg_value = int(arg_value)
                    except ValueError:
                        pass

                args[arg_name] = arg_value

        return tool_name, args


class ToolManager:
    """Manager for tool execution.

    Handles tool discovery, validation, and execution.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        """Initialize tool manager.

        Args:
            registry: Optional tool registry (creates default if None)
        """
        self.registry = registry or ToolRegistry()
        self._execution_log: List[Dict[str, Any]] = []

    def execute(
        self,
        tool_name: str,
        **kwargs
    ) -> ToolResult:
        """Execute a tool.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool arguments

        Returns:
            Tool execution result
        """
        tool = self.registry.get_tool(tool_name)

        if tool is None:
            return ToolResult(
                success=False,
                error=f"Tool not found: {tool_name}",
            )

        try:
            # Validate arguments
            self._validate_arguments(tool, kwargs)

            # Execute tool
            output = tool.function(**kwargs)

            # Log execution
            self._log_execution(tool_name, kwargs, output, None)

            return ToolResult(
                success=True,
                output=output,
                metadata={"tool_name": tool_name},
            )

        except Exception as e:
            self._log_execution(tool_name, kwargs, None, str(e))

            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}",
                metadata={"tool_name": tool_name},
            )

    def _validate_arguments(
        self,
        tool: ToolDefinition,
        kwargs: Dict[str, Any]
    ) -> None:
        """Validate tool arguments.

        Args:
            tool: Tool definition
            kwargs: Provided arguments

        Raises:
            ValueError: If arguments are invalid
        """
        # Check required parameters
        for param_name, param_info in tool.parameters.items():
            if param_info.get("default") is None and param_name not in kwargs:
                raise ValueError(f"Missing required parameter: {param_name}")

        # Check for unknown parameters
        unknown_params = set(kwargs.keys()) - set(tool.parameters.keys())
        if unknown_params:
            raise ValueError(f"Unknown parameters: {unknown_params}")

    def _log_execution(
        self,
        tool_name: str,
        args: Dict[str, Any],
        output: Any,
        error: Optional[str]
    ) -> None:
        """Log tool execution.

        Args:
            tool_name: Tool name
            args: Tool arguments
            output: Tool output
            error: Error message if any
        """
        self._execution_log.append({
            "tool": tool_name,
            "args": args,
            "output": output,
            "error": error,
        })

    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get execution history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of execution records
        """
        return self._execution_log[-limit:]

    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_log = []

    def parse_and_execute(self, text: str) -> Optional[ToolResult]:
        """Parse a tool call from text and execute.

        Args:
            text: Text containing tool call

        Returns:
            Tool result or None if no tool call found
        """
        parsed = self.registry.parse_tool_call(text)

        if parsed is None:
            return None

        tool_name, args = parsed
        return self.execute(tool_name, **args)


# Default global registry
_default_registry = ToolRegistry()


def register_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: str = "general",
    examples: Optional[List[str]] = None,
) -> Callable:
    """Register a tool in the default registry.

    Args:
        name: Tool name
        description: Tool description
        category: Tool category
        examples: Usage examples

    Returns:
        Decorator function
    """
    return _default_registry.register(
        name=name,
        description=description,
        category=category,
        examples=examples,
    )


def get_default_registry() -> ToolRegistry:
    """Get the default tool registry.

    Returns:
        Default registry instance
    """
    return _default_registry
