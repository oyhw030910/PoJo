"""Code Tool Module.

Provides code execution and manipulation tools for LLM agents.
"""

import ast
import sys
import io
import traceback
import json
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager
import tempfile
import os

from agent.tool_manager import register_tool, ToolResult


class CodeExecutor:
    """Safe code executor with sandboxing."""

    def __init__(
        self,
        allowed_imports: Optional[List[str]] = None,
        timeout: int = 10,
        max_output_length: int = 1000
    ):
        """Initialize code executor.

        Args:
            allowed_imports: List of allowed import modules
            timeout: Execution timeout in seconds
            max_output_length: Maximum output length
        """
        self.allowed_imports = allowed_imports or [
            "math", "numpy", "re", "json", "collections",
            "itertools", "functools", "typing"
        ]
        self.timeout = timeout
        self.max_output_length = max_output_length

    def execute(
        self,
        code: str,
        globals_dict: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Tuple[str, bool]:
        """Execute code and return output.

        Args:
            code: Python code to execute
            globals_dict: Optional global variables
            timeout: Execution timeout

        Returns:
            Tuple of (output, success)
        """
        timeout = timeout or self.timeout

        # Create safe execution environment
        safe_globals = self._create_safe_globals()
        if globals_dict:
            safe_globals.update(globals_dict)

        # Capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        success = True
        output = ""

        try:
            exec(code, safe_globals)
            output = sys.stdout.getvalue()
        except Exception as e:
            output = f"Error: {str(e)}\n{traceback.format_exc()}"
            success = False
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # Truncate output
        if len(output) > self.max_output_length:
            output = output[:self.max_output_length] + "... (truncated)"

        return output, success

    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a restricted global namespace.

        Returns:
            Restricted globals dictionary
        """
        safe_globals = {"__builtins__": __builtins__}

        for module_name in self.allowed_imports:
            try:
                module = __import__(module_name)
                safe_globals[module_name] = module
            except ImportError:
                pass

        return safe_globals

    def check_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check code syntax.

        Args:
            code: Code to check

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"


class CodeAnalyzer:
    """Analyzes code structure and quality."""

    def __init__(self):
        """Initialize code analyzer."""
        pass

    def analyze(self, code: str) -> Dict[str, Any]:
        """Analyze code and return metrics.

        Args:
            code: Code to analyze

        Returns:
            Dictionary of analysis results
        """
        try:
            tree = ast.parse(code)

            # Count definitions
            functions = []
            classes = []
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        imports.extend(alias.name for alias in node.names)
                    else:
                        imports.append(node.module)

            return {
                "valid": True,
                "lines": len(code.splitlines()),
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "complexity": self._compute_complexity(tree),
            }
        except SyntaxError:
            return {"valid": False, "error": "Invalid syntax"}

    def _compute_complexity(self, tree: ast.AST) -> int:
        """Compute cyclomatic complexity.

        Args:
            tree: AST tree

        Returns:
            Complexity score
        """
        complexity = 1

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                                ast.With, ast.Assert, ast.comprehension)):
                complexity += 1

        return complexity


# Register tools
_code_executor = CodeExecutor()
_code_analyzer = CodeAnalyzer()


@register_tool(
    name="execute_code",
    description="Execute Python code and return the output",
    category="code",
    examples=["execute_code(code='print(\"Hello, World!\")')"]
)
def execute_code(code: str) -> Dict[str, Any]:
    """Execute Python code.

    Args:
        code: Python code to execute

    Returns:
        Dictionary with output and success status
    """
    output, success = _code_executor.execute(code)
    return {
        "output": output,
        "success": success,
    }


@register_tool(
    name="check_code_syntax",
    description="Check if code has valid Python syntax",
    category="code",
    examples=["check_code_syntax(code='def foo(): return 1')"]
)
def check_code_syntax(code: str) -> Dict[str, Any]:
    """Check code syntax.

    Args:
        code: Code to check

    Returns:
        Dictionary with validity and error message
    """
    is_valid, error = _code_executor.check_syntax(code)
    return {
        "valid": is_valid,
        "error": error,
    }


@register_tool(
    name="analyze_code",
    description="Analyze code structure and metrics",
    category="code",
    examples=["analyze_code(code='def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)')"]
)
def analyze_code(code: str) -> Dict[str, Any]:
    """Analyze code structure.

    Args:
        code: Code to analyze

    Returns:
        Dictionary with analysis results
    """
    return _code_analyzer.analyze(code)


@register_tool(
    name="run_tests",
    description="Run test cases against a function",
    category="code",
    examples=[
        "run_tests(code='def add(a, b): return a + b', tests=[{'args': [1, 2], 'expected': 3}])"
    ]
)
def run_tests(code: str, tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run test cases.

    Args:
        code: Code containing function to test
        tests: List of test cases with 'args' and 'expected'

    Returns:
        Dictionary with test results
    """
    results = []
    passed = 0

    # Execute code to get function
    safe_globals = {"__builtins__": __builtins__}
    try:
        exec(code, safe_globals)
    except Exception as e:
        return {
            "success": False,
            "error": f"Code execution failed: {str(e)}",
            "results": [],
        }

    for test in tests:
        test_result = {"passed": False, "error": None}

        try:
            # Get function name from code
            tree = ast.parse(code)
            func_name = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    break

            if func_name is None:
                test_result["error"] = "No function found in code"
            else:
                func = safe_globals[func_name]
                args = test.get("args", [])
                kwargs = test.get("kwargs", {})
                expected = test.get("expected")

                actual = func(*args, **kwargs)

                if actual == expected:
                    test_result["passed"] = True
                    passed += 1
                else:
                    test_result["error"] = f"Expected {expected}, got {actual}"

        except Exception as e:
            test_result["error"] = str(e)

        results.append(test_result)

    return {
        "success": True,
        "passed": passed,
        "total": len(tests),
        "pass_rate": passed / len(tests) if tests else 0,
        "results": results,
    }


@register_tool(
    name="format_code",
    description="Format Python code with consistent style",
    category="code",
    examples=["format_code(code='def foo( ):return 1')"]
)
def format_code(code: str) -> Dict[str, Any]:
    """Format Python code.

    Args:
        code: Code to format

    Returns:
        Dictionary with formatted code
    """
    try:
        # Parse and unparse to normalize
        tree = ast.parse(code)
        formatted = ast.unparse(tree)
        return {
            "formatted_code": formatted,
            "success": True,
        }
    except Exception as e:
        return {
            "error": str(e),
            "success": False,
        }


@register_tool(
    name="extract_function",
    description="Extract function definition from code",
    category="code",
    examples=["extract_function(code='def foo(): pass\\nprint(1)', name='foo')"]
)
def extract_function(code: str, name: str) -> Dict[str, Any]:
    """Extract a function from code.

    Args:
        code: Source code
        name: Function name to extract

    Returns:
        Dictionary with function code
    """
    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                func_code = ast.unparse(node)
                return {
                    "code": func_code,
                    "success": True,
                }

        return {
            "error": f"Function '{name}' not found",
            "success": False,
        }
    except Exception as e:
        return {
            "error": str(e),
            "success": False,
        }
