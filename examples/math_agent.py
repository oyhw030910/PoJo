"""Math Agent - Specialized Agent for Mathematical Reasoning.

This agent is specialized for solving mathematical problems
step by step with reasoning.
"""

import torch
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from agent.llm_wrapper import LLMWrapper, LLMConfig
from agent.memory import MemoryManager
from environment.math_env import MathEnvironment, MathTask
from tools.search_tool import calculate


@dataclass
class MathAgentConfig:
    """Configuration for Math Agent.

    Attributes:
        model_name: Model to use
        device: Device
        use_calculator: Whether to use calculator tool
        step_by_step: Whether to show step-by-step reasoning
        max_steps: Maximum reasoning steps
    """
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "auto"  # Auto-detect: cuda > mps > cpu
    use_calculator: bool = True
    step_by_step: bool = True
    max_steps: int = 10


class MathAgent:
    """Agent specialized for mathematical reasoning.

    Features:
    - Step-by-step reasoning
    - Calculator integration
    - Answer verification
    - Multiple solving strategies

    Usage:
        agent = MathAgent()
        result = agent.solve("What is 15% of 80?")
    """

    def __init__(self, config: Optional[MathAgentConfig] = None):
        """Initialize math agent.

        Args:
            config: Agent configuration
        """
        self.config = config or MathAgentConfig()
        self.device = torch.device(self.config.device)

        # Initialize LLM
        self._init_llm()

        # State
        self._current_problem: Optional[str] = None
        self._reasoning_steps: List[str] = []
        self._calculator_uses: int = 0

    def _init_llm(self) -> None:
        """Initialize the LLM."""
        llm_config = LLMConfig(
            model_name=self.config.model_name,
            device=self.config.device,
        )
        self.llm = LLMWrapper(llm_config)
        print(f"Math Agent initialized: {self.config.model_name}")

    def solve(
        self,
        problem: str,
        topic: str = "general",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Solve a mathematical problem.

        Args:
            problem: Problem description
            topic: Problem topic (arithmetic, algebra, geometry, etc.)
            verbose: Whether to print progress

        Returns:
            Solution results
        """
        self._current_problem = problem
        self._reasoning_steps = []
        self._calculator_uses = 0

        if verbose:
            print(f"\n{'='*50}")
            print(f"Problem: {problem}")
            print(f"Topic: {topic}")
            print(f"{'='*50}\n")

        # Solve based on topic
        if topic == "arithmetic" or self._is_arithmetic(problem):
            result = self._solve_arithmetic(problem, verbose)
        elif topic == "algebra" or self._is_algebra(problem):
            result = self._solve_algebra(problem, verbose)
        elif topic == "geometry" or self._is_geometry(problem):
            result = self._solve_geometry(problem, verbose)
        elif topic == "percentage" or self._is_percentage(problem):
            result = self._solve_percentage(problem, verbose)
        else:
            result = self._solve_general(problem, verbose)

        if verbose:
            print(f"\n{'='*50}")
            print(f"Answer: {result.get('answer', 'N/A')}")
            print(f"{'='*50}\n")

        return result

    def _is_arithmetic(self, problem: str) -> bool:
        """Check if problem is arithmetic."""
        arithmetic_keywords = [
            "add", "subtract", "multiply", "divide",
            "sum", "difference", "product", "quotient",
            "plus", "minus", "times", "divided by",
        ]
        return any(kw in problem.lower() for kw in arithmetic_keywords)

    def _is_algebra(self, problem: str) -> bool:
        """Check if problem is algebra."""
        algebra_keywords = [
            "solve for", "equation", "variable",
            "x =", "find x", "find y",
        ]
        return any(kw in problem.lower() for kw in algebra_keywords)

    def _is_geometry(self, problem: str) -> bool:
        """Check if problem is geometry."""
        geometry_keywords = [
            "area", "perimeter", "volume", "circle",
            "triangle", "rectangle", "radius", "diameter",
        ]
        return any(kw in problem.lower() for kw in geometry_keywords)

    def _is_percentage(self, problem: str) -> bool:
        """Check if problem involves percentages."""
        return "%" in problem or "percent" in problem.lower()

    def _solve_arithmetic(
        self,
        problem: str,
        verbose: bool
    ) -> Dict[str, Any]:
        """Solve arithmetic problem.

        Args:
            problem: Problem description
            verbose: Whether to print

        Returns:
            Solution
        """
        if verbose:
            print("Strategy: Arithmetic calculation\n")

        # Try to extract numbers and operations
        numbers = self._extract_numbers(problem)
        operation = self._identify_operation(problem)

        if verbose:
            print(f"Extracted numbers: {numbers}")
            print(f"Operation: {operation}\n")

        if numbers and operation:
            # Use calculator
            expression = self._build_expression(numbers, operation)
            if verbose:
                print(f"Calculating: {expression}")

            calc_result = calculate(expression)

            self._calculator_uses += 1
            self._reasoning_steps.append(f"Calculate: {expression} = {calc_result.get('result')}")

            return {
                "answer": calc_result.get("result"),
                "steps": self._reasoning_steps.copy(),
                "calculator_uses": self._calculator_uses,
                "method": "calculator",
            }

        # Fallback to LLM
        return self._solve_with_llm(problem, verbose)

    def _solve_percentage(
        self,
        problem: str,
        verbose: bool
    ) -> Dict[str, Any]:
        """Solve percentage problem.

        Args:
            problem: Problem description
            verbose: Whether to print

        Returns:
            Solution
        """
        if verbose:
            print("Strategy: Percentage calculation\n")

        # Extract percentage and base number
        percent_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", problem, re.IGNORECASE)
        base_match = re.search(r"(?:of|from)\s*(\d+(?:\.\d+)?)", problem, re.IGNORECASE)

        if percent_match and base_match:
            percent = float(percent_match.group(1))
            base = float(base_match.group(1))

            expression = f"{percent}/100 * {base}"
            calc_result = calculate(expression)

            self._calculator_uses += 1
            self._reasoning_steps.append(f"Convert {percent}% to decimal: {percent/100}")
            self._reasoning_steps.append(f"Calculate: {percent/100} × {base} = {calc_result.get('result')}")

            return {
                "answer": calc_result.get("result"),
                "steps": self._reasoning_steps.copy(),
                "calculator_uses": self._calculator_uses,
                "method": "percentage_formula",
            }

        return self._solve_with_llm(problem, verbose)

    def _solve_algebra(
        self,
        problem: str,
        verbose: bool
    ) -> Dict[str, Any]:
        """Solve algebra problem.

        Args:
            problem: Problem description
            verbose: Whether to print

        Returns:
            Solution
        """
        if verbose:
            print("Strategy: Algebraic manipulation\n")

        # Try to extract equation
        equation_pattern = r"(\d*\w?\s*[+\-*/]?\s*\d*\s*=\s*\d*)"
        match = re.search(equation_pattern, problem)

        if match:
            equation = match.group(1)
            self._reasoning_steps.append(f"Equation: {equation}")

            # Ask LLM to solve
            solution_prompt = f"""Solve the equation step by step:
{equation}

Show each step clearly. End with "The answer is: [value]"
"""
            solution = self.llm.generate(
                prompt=solution_prompt,
                max_new_tokens=256,
                temperature=0.3,
            )

            # Extract answer
            answer_match = re.search(r"answer is:\s*(\d+(?:\.\d+)?)", solution, re.IGNORECASE)
            answer = answer_match.group(1) if answer_match else "Could not solve"

            self._reasoning_steps.append(f"Solution: {solution[:200]}")

            return {
                "answer": answer,
                "steps": self._reasoning_steps.copy(),
                "method": "algebraic",
            }

        return self._solve_with_llm(problem, verbose)

    def _solve_geometry(
        self,
        problem: str,
        verbose: bool
    ) -> Dict[str, Any]:
        """Solve geometry problem.

        Args:
            problem: Problem description
            verbose: Whether to print

        Returns:
            Solution
        """
        if verbose:
            print("Strategy: Geometry formula\n")

        # Check for specific geometry problems
        if "area" in problem.lower() and "circle" in problem.lower():
            # Area of circle: πr²
            radius_match = re.search(r"radius\s*(?:of|is)?\s*(\d+(?:\.\d+)?)", problem, re.IGNORECASE)
            if radius_match:
                radius = float(radius_match.group(1))
                expression = f"3.14159 * {radius} * {radius}"
                calc_result = calculate(expression)

                self._calculator_uses += 1
                self._reasoning_steps.append(f"Area = π × r² = 3.14159 × {radius}²")
                self._reasoning_steps.append(f"Area = {calc_result.get('result')}")

                return {
                    "answer": calc_result.get("result"),
                    "steps": self._reasoning_steps.copy(),
                    "calculator_uses": self._calculator_uses,
                    "method": "circle_area",
                }

        if "area" in problem.lower() and "rectangle" in problem.lower():
            length_match = re.search(r"length\s*(?:of|is)?\s*(\d+)", problem, re.IGNORECASE)
            width_match = re.search(r"width\s*(?:of|is)?\s*(\d+)", problem, re.IGNORECASE)
            if length_match and width_match:
                length = int(length_match.group(1))
                width = int(width_match.group(1))
                expression = f"{length} * {width}"
                calc_result = calculate(expression)

                self._calculator_uses += 1
                self._reasoning_steps.append(f"Area = length × width = {length} × {width}")
                self._reasoning_steps.append(f"Area = {calc_result.get('result')}")

                return {
                    "answer": calc_result.get("result"),
                    "steps": self._reasoning_steps.copy(),
                    "calculator_uses": self._calculator_uses,
                    "method": "rectangle_area",
                }

        return self._solve_with_llm(problem, verbose)

    def _solve_general(
        self,
        problem: str,
        verbose: bool
    ) -> Dict[str, Any]:
        """Solve general math problem using LLM.

        Args:
            problem: Problem description
            verbose: Whether to print

        Returns:
            Solution
        """
        return self._solve_with_llm(problem, verbose)

    def _solve_with_llm(
        self,
        problem: str,
        verbose: bool
    ) -> Dict[str, Any]:
        """Solve problem using LLM reasoning.

        Args:
            problem: Problem description
            verbose: Whether to print

        Returns:
            Solution
        """
        if verbose:
            print("Strategy: LLM reasoning\n")

        prompt = f"""Solve this math problem step by step:

Problem: {problem}

Instructions:
1. Show your reasoning clearly
2. Show each calculation step
3. End with "The answer is: [value]"

Solution:
"""

        solution = self.llm.generate(
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.3,
        )

        if verbose:
            print(f"Reasoning:\n{solution}\n")

        self._reasoning_steps.append(f"LLM reasoning: {solution[:300]}")

        # Extract answer
        answer_match = re.search(r"answer is:\s*(\d+(?:\.\d+)?)", solution, re.IGNORECASE)
        answer = answer_match.group(1) if answer_match else "Could not determine"

        return {
            "answer": answer,
            "steps": self._reasoning_steps.copy(),
            "method": "llm_reasoning",
            "full_solution": solution,
        }

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text.

        Args:
            text: Input text

        Returns:
            List of numbers
        """
        matches = re.findall(r"\d+(?:\.\d+)?", text)
        return [float(m) for m in matches]

    def _identify_operation(self, text: str) -> Optional[str]:
        """Identify mathematical operation from text.

        Args:
            text: Input text

        Returns:
            Operation symbol (+, -, *, /)
        """
        text_lower = text.lower()

        if any(kw in text_lower for kw in ["add", "sum", "plus", "total", "combined"]):
            return "+"
        elif any(kw in text_lower for kw in ["subtract", "difference", "minus", "remain"]):
            return "-"
        elif any(kw in text_lower for kw in ["multiply", "product", "times", "double", "triple"]):
            return "*"
        elif any(kw in text_lower for kw in ["divide", "quotient", "per", "ratio", "split"]):
            return "/"

        return None

    def _build_expression(
        self,
        numbers: List[float],
        operation: str
    ) -> str:
        """Build mathematical expression.

        Args:
            numbers: Numbers to use
            operation: Operation symbol

        Returns:
            Expression string
        """
        return operation.join(str(n) for n in numbers)

    def verify_answer(
        self,
        problem: str,
        answer: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Verify if an answer is correct.

        Args:
            problem: Original problem
            answer: Answer to verify
            verbose: Whether to print

        Returns:
            Verification results
        """
        prompt = f"""Verify if this answer is correct for the given problem.

Problem: {problem}
Answer: {answer}

Explain your reasoning and state whether the answer is correct or incorrect."""

        verification = self.llm.generate(
            prompt=prompt,
            max_new_tokens=256,
            temperature=0.3,
        )

        if verbose:
            print(f"\nVerification:\n{verification}\n")

        # Check for correctness indicators
        is_correct = any(
            kw in verification.lower()
            for kw in ["correct", "right", "yes", "true"]
        ) and not any(
            kw in verification.lower()
            for kw in ["incorrect", "wrong", "no", "false", "error"]
        )

        return {
            "is_correct": is_correct,
            "explanation": verification,
        }

    def get_reasoning_steps(self) -> List[str]:
        """Get the reasoning steps from last problem.

        Returns:
            List of reasoning steps
        """
        return self._reasoning_steps.copy()


# Example usage
if __name__ == "__main__":
    # Create math agent
    agent = MathAgent()

    # Example 1: Arithmetic
    print("\n" + "="*60)
    print("Example 1: Arithmetic")
    print("="*60)

    agent.solve("What is 25 + 17 + 38?", topic="arithmetic")

    # Example 2: Percentage
    print("\n" + "="*60)
    print("Example 2: Percentage")
    print("="*60)

    agent.solve("What is 15% of 80?", topic="percentage")

    # Example 3: Geometry
    print("\n" + "="*60)
    print("Example 3: Geometry")
    print("="*60)

    agent.solve("What is the area of a rectangle with length 8 and width 5?", topic="geometry")

    # Example 4: General
    print("\n" + "="*60)
    print("Example 4: General Reasoning")
    print("="*60)

    agent.solve("A train travels at 60 mph for 2.5 hours. How far does it travel?")
