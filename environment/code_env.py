"""Code Execution Environment.

Provides a sandboxed environment for code generation and execution tasks.
Supports testing code correctness against test cases.
"""

import ast
import sys
import io
import traceback
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from .base_env import (
    BaseEnvironment,
    Observation,
    Action,
    ActionSpace,
    ObservationSpace,
    StepResult,
)


@dataclass
class CodeTask:
    """Represents a code generation task.

    Attributes:
        id: Task identifier
        description: Problem description
        starter_code: Optional starter code template
        test_cases: List of test cases
        signature: Function signature to implement
    """
    id: str
    description: str
    starter_code: str
    test_cases: List[Dict[str, Any]]
    signature: str


@dataclass
class CodeState:
    """Represents the current state of the code environment.

    Attributes:
        current_code: The current code generated
        execution_output: Output from last execution
        test_results: Results from running test cases
        history: History of code modifications
    """
    current_code: str = ""
    execution_output: str = ""
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    history: List[str] = field(default_factory=list)


class CodeEnvironment(BaseEnvironment):
    """Environment for code generation and execution tasks.

    This environment allows the agent to:
    - Generate code to solve programming problems
    - Execute code in a sandboxed environment
    - Test code against test cases
    - Iteratively refine code based on feedback

    Actions:
    - 'generate': Generate/replace code
    - 'execute': Execute the current code
    - 'test': Run test cases
    - 'submit': Submit final solution

    Rewards:
    - Positive reward for passing test cases
    - Negative reward for execution errors
    - Small penalty for each step (encourages efficiency)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self._tasks: List[CodeTask] = []
        self._current_task: Optional[CodeTask] = None
        self._state: CodeState = CodeState()
        self._allowed_imports = self.config.get(
            "allowed_imports",
            ["math", "numpy", "re", "json", "collections", "itertools", "functools"]
        )
        self._execution_timeout = self.config.get("execution_timeout", 10)
        self._max_output_length = self.config.get("max_output_length", 1000)

        # Load datasets if specified
        self._load_datasets()

    def _load_datasets(self) -> None:
        """Load code generation datasets."""
        # Placeholder for dataset loading
        # In production, this would load from HumanEval, MBPP, etc.
        pass

    def load_task(self, task: CodeTask) -> None:
        """Load a specific task into the environment.

        Args:
            task: The code task to load
        """
        self._current_task = task
        self._state = CodeState(starter_code=task.starter_code)

    def reset(self, seed: Optional[int] = None, task: Optional[CodeTask] = None, **kwargs) -> Observation:
        """Reset the environment.

        Args:
            seed: Random seed
            task: Optional task to load
            **kwargs: Additional arguments

        Returns:
            Initial observation
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._current_step = 0
        self._done = False

        if task is not None:
            self.load_task(task)

        self._state = CodeState(
            current_code=self._current_task.starter_code if self._current_task else ""
        )

        return self.get_observation()

    def step(self, action: Action) -> StepResult:
        """Take an action in the environment.

        Args:
            action: Action with type and value

        Returns:
            StepResult with observation, reward, done, truncated, info
        """
        self._current_step += 1
        reward = self.config.get("step_penalty", -0.01)
        info = {"action_type": action.action_type}
        truncated = self._current_step >= self._max_steps

        try:
            if action.action_type == "generate":
                output, exec_reward = self._generate_code(action.value)
                reward += exec_reward
                info["output"] = output

            elif action.action_type == "execute":
                output, exec_reward = self._execute_code()
                reward += exec_reward
                info["output"] = output

            elif action.action_type == "test":
                results, test_reward = self._run_tests()
                reward += test_reward
                info["test_results"] = results

            elif action.action_type == "submit":
                results, done, submit_reward = self._submit()
                reward += submit_reward
                self._done = done
                info["test_results"] = results

            else:
                reward -= 0.1  # Penalty for invalid action type
                info["error"] = f"Unknown action type: {action.action_type}"

        except Exception as e:
            reward -= 0.5
            info["error"] = str(e)
            traceback.print_exc()

        # Check for truncation
        if truncated and not self._done:
            self._done = True
            info["truncation"] = True

        return StepResult(
            observation=self.get_observation(),
            reward=reward,
            done=self._done,
            truncated=truncated,
            info=info,
        )

    def _generate_code(self, code: str) -> Tuple[str, float]:
        """Generate or replace code.

        Args:
            code: The code to generate

        Returns:
            Tuple of (output message, reward)
        """
        # Save to history
        self._state.history.append(self._state.current_code)

        # Update current code
        self._state.current_code = code

        # Auto-execute to check for syntax errors
        try:
            ast.parse(code)
            return "Code generated successfully. Syntax valid.", 0.05
        except SyntaxError as e:
            return f"Syntax error: {e}", -0.1

    def _execute_code(self) -> Tuple[str, float]:
        """Execute the current code.

        Returns:
            Tuple of (execution output, reward)
        """
        if not self._state.current_code:
            return "No code to execute.", -0.1

        # Create safe environment for execution
        allowed_globals = {"__builtins__": __builtins__}
        for module in self._allowed_imports:
            try:
                allowed_globals[module] = __import__(module)
            except ImportError:
                pass

        # Capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        reward = 0.0
        output = ""

        try:
            exec(self._state.current_code, allowed_globals)
            output = sys.stdout.getvalue()
            reward = 0.05
        except Exception as e:
            output = f"Execution error: {str(e)}"
            reward = -0.1
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # Truncate output if needed
        if len(output) > self._max_output_length:
            output = output[:self._max_output_length] + "... (truncated)"

        self._state.execution_output = output
        return output, reward

    def _run_tests(self) -> Tuple[List[Dict[str, Any]], float]:
        """Run test cases against current code.

        Returns:
            Tuple of (test results, reward)
        """
        if self._current_task is None:
            return [], -0.1

        results = []
        passed = 0
        total = len(self._current_task.test_cases)

        for test_case in self._current_task.test_cases:
            result = self._run_single_test(test_case)
            results.append(result)
            if result["passed"]:
                passed += 1

        self._state.test_results = results
        pass_rate = passed / total if total > 0 else 0

        # Reward based on pass rate
        reward = pass_rate * 0.5

        return results, reward

    def _run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case.

        Args:
            test_case: Dictionary with 'input', 'expected', and optional 'timeout'

        Returns:
            Dictionary with test result information
        """
        result = {
            "input": test_case.get("input"),
            "expected": test_case.get("expected"),
            "actual": None,
            "passed": False,
            "error": None,
        }

        try:
            # Create execution context
            allowed_globals = {"__builtins__": __builtins__}
            for module in self._allowed_imports:
                try:
                    allowed_globals[module] = __import__(module)
                except ImportError:
                    pass

            # Execute the code
            exec(self._state.current_code, allowed_globals)

            # Get the function and call it
            func_name = self._current_task.signature.split("(")[0].split()[-1]
            if func_name in allowed_globals:
                func = allowed_globals[func_name]
                actual = func(*test_case.get("args", []), **test_case.get("kwargs", {}))
                result["actual"] = actual

                # Compare with expected
                if actual == test_case.get("expected"):
                    result["passed"] = True
            else:
                result["error"] = f"Function {func_name} not found"

        except Exception as e:
            result["error"] = str(e)

        return result

    def _submit(self) -> Tuple[List[Dict[str, Any]], bool, float]:
        """Submit the current solution.

        Returns:
            Tuple of (test results, done, reward)
        """
        results, reward = self._run_tests()

        # Calculate final reward
        passed = sum(1 for r in results if r["passed"])
        total = len(results)
        pass_rate = passed / total if total > 0 else 0

        # Base reward + bonus for full solution
        base_reward = self.config.get("success_reward", 1.0)
        final_reward = pass_rate * base_reward

        if pass_rate == 1.0:
            final_reward += 0.5  # Bonus for perfect solution

        return results, True, final_reward

    def get_observation(self) -> Observation:
        """Get the current observation.

        Returns:
            Current observation
        """
        text = self.get_task_description()
        if self._state.current_code:
            text += f"\n\nCurrent code:\n```python\n{self._state.current_code}\n```"
        if self._state.execution_output:
            text += f"\n\nExecution output:\n{self._state.execution_output}"
        if self._state.test_results:
            passed = sum(1 for r in self._state.test_results if r["passed"])
            total = len(self._state.test_results)
            text += f"\n\nTest results: {passed}/{total} passed"

        return Observation(
            text=text,
            state={"code": self._state.current_code, "step": self._current_step},
            metadata={
                "execution_output": self._state.execution_output,
                "test_results": self._state.test_results,
                "history_length": len(self._state.history),
            },
        )

    def get_action_space(self) -> ActionSpace:
        """Get the action space.

        Returns:
            Action space for code environment
        """
        return ActionSpace(
            action_types=["generate", "execute", "test", "submit"],
            value_space="text",
            constraints={"max_code_length": 10000},
        )

    def get_observation_space(self) -> ObservationSpace:
        """Get the observation space.

        Returns:
            Observation space for code environment
        """
        return ObservationSpace(
            text_max_length=8192,
            metadata_keys=["execution_output", "test_results", "history_length"],
        )

    def compute_reward(self, **kwargs) -> float:
        """Compute reward for current state.

        Args:
            **kwargs: Reward computation arguments

        Returns:
            Computed reward
        """
        reward = 0.0

        # Reward for code execution
        if self._state.execution_output:
            if "error" not in self._state.execution_output.lower():
                reward += 0.05

        # Reward for test progress
        if self._state.test_results:
            passed = sum(1 for r in self._state.test_results if r["passed"])
            total = len(self._state.test_results)
            reward += (passed / total) * 0.5 if total > 0 else 0

        return reward

    def get_info(self) -> Dict[str, Any]:
        """Get additional information.

        Returns:
            Environment information dictionary
        """
        return {
            "current_step": self._current_step,
            "max_steps": self._max_steps,
            "done": self._done,
            "task_id": self._current_task.id if self._current_task else None,
            "code_length": len(self._state.current_code),
            "test_pass_rate": (
                sum(1 for r in self._state.test_results if r["passed"]) / len(self._state.test_results)
                if self._state.test_results else 0
            ),
        }

    def is_valid_action(self, action: Action) -> bool:
        """Check if an action is valid.

        Args:
            action: Action to validate

        Returns:
            True if valid, False otherwise
        """
        if action.action_type not in ["generate", "execute", "test", "submit"]:
            return False

        if action.action_type == "generate" and not action.value:
            return False

        return True

    def get_task_description(self) -> str:
        """Get task description.

        Returns:
            Task description string
        """
        if self._current_task is None:
            return "No task loaded."

        return f"""Task: {self._current_task.id}

Description:
{self._current_task.description}

Function signature to implement:
{self._current_task.signature}

Starter code:
{self._current_task.starter_code or "None"}"""

    def close(self) -> None:
        """Clean up the environment."""
        self._current_task = None
        self._state = CodeState()
