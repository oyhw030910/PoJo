"""Code Agent - Specialized Agent for Code Generation.

This agent is specialized for code generation and debugging tasks.
"""

import torch
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from agent.llm_wrapper import LLMWrapper, LLMConfig
from agent.memory import MemoryManager
from environment.code_env import CodeEnvironment, CodeTask
from tools.code_tool import execute_code, check_code_syntax, analyze_code, run_tests


@dataclass
class CodeAgentConfig:
    """Configuration for Code Agent.

    Attributes:
        model_name: Model to use
        device: Device
        max_iterations: Maximum code refinement iterations
        auto_test: Whether to auto-test generated code
        use_lora: Whether to use LoRA
    """
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "auto"  # Auto-detect: cuda > mps > cpu
    max_iterations: int = 5
    auto_test: bool = True
    use_lora: bool = False


class CodeAgent:
    """Agent specialized for code generation tasks.

    Features:
    - Generate code from description
    - Auto-test with test cases
    - Iterative refinement
    - Syntax checking
    - Code analysis

    Usage:
        agent = CodeAgent()
        result = agent.generate_code(
            description="Write a function to reverse a string",
            test_cases=[{"args": ["hello"], "expected": "olleh"}]
        )
    """

    def __init__(self, config: Optional[CodeAgentConfig] = None):
        """Initialize code agent.

        Args:
            config: Agent configuration
        """
        self.config = config or CodeAgentConfig()
        self.device = torch.device(self.config.device)

        # Initialize LLM
        self._init_llm()

        # Code-specific state
        self._code_history: List[str] = []
        self._test_results: List[Dict] = []

    def _init_llm(self) -> None:
        """Initialize the LLM for code generation."""
        llm_config = LLMConfig(
            model_name=self.config.model_name,
            lora_enabled=self.config.use_lora,
            device=self.config.device,
        )
        self.llm = LLMWrapper(llm_config)
        print(f"Code Agent initialized: {self.config.model_name}")

    def generate_code(
        self,
        description: str,
        test_cases: Optional[List[Dict[str, Any]]] = None,
        starter_code: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Generate code from description.

        Args:
            description: Function description
            test_cases: Optional test cases
            starter_code: Optional starter code template
            verbose: Whether to print progress

        Returns:
            Generation results
        """
        if verbose:
            print(f"\n{'='*50}")
            print(f"Task: {description}")
            print(f"{'='*50}\n")

        # Build prompt
        prompt = self._build_generation_prompt(description, starter_code)

        if verbose:
            print("Generating code...")

        # Generate initial code
        generated_code = self.llm.generate(
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.3,  # Lower temperature for code
        )

        # Extract code from response
        code = self._extract_code(generated_code)
        self._code_history.append(code)

        if verbose:
            print(f"\nGenerated code:\n```python\n{code}\n```\n")

        # Check syntax
        syntax_result = self._check_syntax(code)
        if not syntax_result["valid"]:
            if verbose:
                print(f"Syntax error: {syntax_result['error']}")
            return self._refine_code(
                code, description, syntax_result["error"], test_cases, verbose
            )

        # Run tests if provided
        if self.config.auto_test and test_cases:
            test_result = self._run_tests(code, test_cases)
            if verbose:
                passed = test_result.get("passed", 0)
                total = test_result.get("total", len(test_cases))
                print(f"Tests: {passed}/{total} passed")

            if passed < total:
                return self._refine_code(
                    code, description, "Tests failed", test_cases, verbose,
                    test_result=test_result
                )

        return {
            "success": True,
            "code": code,
            "syntax_valid": syntax_result["valid"],
            "test_results": test_result if test_cases else None,
            "iterations": 1,
        }

    def _build_generation_prompt(
        self,
        description: str,
        starter_code: Optional[str] = None
    ) -> str:
        """Build code generation prompt.

        Args:
            description: Task description
            starter_code: Optional starter code

        Returns:
            Prompt string
        """
        prompt = f"""Write Python code for the following task:

Task: {description}

Requirements:
- Write clean, efficient code
- Include docstring
- Handle edge cases
- Only output the code, no explanation

"""
        if starter_code:
            prompt += f"Starter code:\n{starter_code}\n\nComplete the function:\n"
        else:
            prompt += "Your code:\n"

        return prompt

    def _extract_code(self, text: str) -> str:
        """Extract code from generated text.

        Args:
            text: Generated text

        Returns:
            Extracted code
        """
        # Try to extract code between markdown fences
        if "```python" in text:
            start = text.find("```python") + len("```python")
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        return text.strip()

    def _check_syntax(self, code: str) -> Dict[str, Any]:
        """Check code syntax.

        Args:
            code: Code to check

        Returns:
            Syntax check result
        """
        try:
            import ast
            ast.parse(code)
            return {"valid": True, "error": None}
        except SyntaxError as e:
            return {"valid": False, "error": str(e)}

    def _run_tests(
        self,
        code: str,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run test cases against code.

        Args:
            code: Code to test
            test_cases: Test cases

        Returns:
            Test results
        """
        passed = 0
        results = []

        # Create execution context
        safe_globals = {"__builtins__": __builtins__}

        try:
            exec(code, safe_globals)

            # Get function name
            import ast
            tree = ast.parse(code)
            func_name = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    break

            if func_name and func_name in safe_globals:
                func = safe_globals[func_name]

                for test in test_cases:
                    try:
                        args = test.get("args", [])
                        kwargs = test.get("kwargs", {})
                        expected = test.get("expected")

                        actual = func(*args, **kwargs)

                        if actual == expected:
                            passed += 1
                            results.append({"passed": True})
                        else:
                            results.append({
                                "passed": False,
                                "error": f"Expected {expected}, got {actual}"
                            })
                    except Exception as e:
                        results.append({"passed": False, "error": str(e)})

        except Exception as e:
            return {
                "passed": 0,
                "total": len(test_cases),
                "error": str(e),
            }

        return {
            "passed": passed,
            "total": len(test_cases),
            "results": results,
        }

    def _refine_code(
        self,
        code: str,
        description: str,
        error: str,
        test_cases: Optional[List[Dict]],
        verbose: bool,
        test_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Refine code based on errors.

        Args:
            code: Current code
            description: Task description
            error: Error message
            test_cases: Test cases
            verbose: Whether to print progress
            test_result: Optional test results

        Returns:
            Refinement results
        """
        iterations = 1

        while iterations < self.config.max_iterations:
            iterations += 1

            if verbose:
                print(f"\nRefining code (iteration {iterations})...")

            # Build refinement prompt
            prompt = self._build_refinement_prompt(
                code, description, error, test_result
            )

            new_code = self.llm.generate(
                prompt=prompt,
                max_new_tokens=512,
                temperature=0.3,
            )

            new_code = self._extract_code(new_code)

            if verbose:
                print(f"\nRefined code:\n```python\n{new_code}\n```\n")

            # Check syntax
            syntax_result = self._check_syntax(new_code)
            if not syntax_result["valid"]:
                code = new_code
                error = syntax_result["error"]
                continue

            # Run tests
            if test_cases:
                test_result = self._run_tests(new_code, test_cases)
                if test_result["passed"] == test_result["total"]:
                    return {
                        "success": True,
                        "code": new_code,
                        "syntax_valid": True,
                        "test_results": test_result,
                        "iterations": iterations,
                    }
                code = new_code
                error = f"Tests: {test_result['passed']}/{test_result['total']} passed"
            else:
                return {
                    "success": True,
                    "code": new_code,
                    "syntax_valid": True,
                    "iterations": iterations,
                }

        return {
            "success": False,
            "code": code,
            "syntax_valid": syntax_result["valid"],
            "error": error,
            "iterations": iterations,
        }

    def _build_refinement_prompt(
        self,
        code: str,
        description: str,
        error: str,
        test_result: Optional[Dict]
    ) -> str:
        """Build code refinement prompt.

        Args:
            code: Current code
            description: Task description
            error: Error message
            test_result: Test results

        Returns:
            Prompt string
        """
        prompt = f"""Fix the following code.

Task: {description}

Current code:
```python
{code}
```

Problem: {error}

"""
        if test_result and "results" in test_result:
            failed = [r for r in test_result["results"] if not r.get("passed")]
            if failed:
                prompt += f"Failed test details: {failed[0].get('error', 'unknown')}\n"

        prompt += """Write the corrected complete code.
Only output the code, no explanation:
"""
        return prompt

    def explain_code(self, code: str, verbose: bool = True) -> str:
        """Explain what code does.

        Args:
            code: Code to explain
            verbose: Whether to print

        Returns:
            Explanation
        """
        prompt = f"""Explain what the following code does:

```python
{code}
```

Provide a clear, concise explanation:"""

        explanation = self.llm.generate(
            prompt=prompt,
            max_new_tokens=256,
            temperature=0.5,
        )

        if verbose:
            print(f"\nExplanation:\n{explanation}")

        return explanation

    def debug_code(
        self,
        code: str,
        error_message: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Debug and fix code.

        Args:
            code: Code to debug
            error_message: Error message
            verbose: Whether to print

        Returns:
            Debug results
        """
        if verbose:
            print(f"\nDebugging code...")
            print(f"Error: {error_message}")

        prompt = f"""Find and fix the bug in the following code:

```python
{code}
```

Error: {error_message}

Provide the fixed code:"""

        fixed_code = self.llm.generate(
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.3,
        )

        fixed_code = self._extract_code(fixed_code)

        if verbose:
            print(f"\nFixed code:\n```python\n{fixed_code}\n```\n")

        return {
            "original_code": code,
            "fixed_code": fixed_code,
            "error": error_message,
        }

    def get_history(self) -> List[str]:
        """Get code generation history.

        Returns:
            List of generated code snippets
        """
        return self._code_history.copy()


# Example usage
if __name__ == "__main__":
    # Create code agent
    agent = CodeAgent()

    # Example 1: Generate code
    print("\n" + "="*60)
    print("Example 1: Generate Code")
    print("="*60)

    result = agent.generate_code(
        description="Write a function that checks if a number is prime",
        test_cases=[
            {"args": [2], "expected": True},
            {"args": [17], "expected": True},
            {"args": [4], "expected": False},
            {"args": [1], "expected": False},
        ],
        verbose=True,
    )

    print(f"\nResult: Success={result['success']}")
    print(f"Iterations: {result['iterations']}")

    # Example 2: Explain code
    print("\n" + "="*60)
    print("Example 2: Explain Code")
    print("="*60)

    sample_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

    agent.explain_code(sample_code)
