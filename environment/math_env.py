"""Math Problem Solving Environment.

Provides an environment for mathematical reasoning and problem solving tasks.
Supports datasets like GSM8K and MATH.
"""

import re
import sympy
from typing import Any, Dict, List, Optional, Tuple, Union
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
class MathTask:
    """Represents a math problem solving task.

    Attributes:
        id: Task identifier
        problem: Problem description
        solution: Ground truth solution
        answer: Final answer
        steps: Optional intermediate solution steps
        topic: Math topic category
    """
    id: str
    problem: str
    solution: str
    answer: str
    steps: List[str] = field(default_factory=list)
    topic: str = "general"


@dataclass
class MathState:
    """Represents the current state of the math environment.

    Attributes:
        current_solution: Current solution attempt
        steps_taken: List of steps taken
        calculator_uses: Number of calculator uses
        history: History of solution attempts
    """
    current_solution: str = ""
    steps_taken: List[str] = field(default_factory=list)
    calculator_uses: int = 0
    history: List[str] = field(default_factory=list)


class MathEnvironment(BaseEnvironment):
    """Environment for mathematical reasoning tasks.

    This environment allows the agent to:
    - Solve mathematical problems step by step
    - Use a calculator for computations
    - Verify answers against ground truth
    - Receive feedback on intermediate steps

    Actions:
    - 'reason': Add reasoning step
    - 'calculate': Use calculator for computation
    - 'verify': Check current answer
    - 'submit': Submit final answer

    Rewards:
    - Positive reward for correct answer
    - Bonus for correct intermediate steps
    - Penalty for incorrect calculations
    - Small step penalty for efficiency
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self._current_task: Optional[MathTask] = None
        self._state: MathState = MathState()
        self._allow_calculator = self.config.get("allow_calculator", True)
        self._calculator_precision = self.config.get("calculator_precision", 10)

        # Dataset storage
        self._datasets: Dict[str, List[MathTask]] = {}

    def load_task(self, task: MathTask) -> None:
        """Load a specific math task.

        Args:
            task: The math task to load
        """
        self._current_task = task
        self._state = MathState()

    def load_dataset(self, name: str, tasks: List[MathTask]) -> None:
        """Load a dataset of math tasks.

        Args:
            name: Dataset name
            tasks: List of math tasks
        """
        self._datasets[name] = tasks

    def sample_task(self, dataset: Optional[str] = None) -> MathTask:
        """Sample a random task.

        Args:
            dataset: Optional dataset name to sample from

        Returns:
            A random math task
        """
        if dataset and dataset in self._datasets:
            tasks = self._datasets[dataset]
        else:
            # Sample from all datasets
            tasks = []
            for ds_tasks in self._datasets.values():
                tasks.extend(ds_tasks)

        if not tasks:
            raise ValueError("No tasks loaded")

        idx = self._rng.integers(0, len(tasks))
        return tasks[idx]

    def reset(self, seed: Optional[int] = None, task: Optional[MathTask] = None, **kwargs) -> Observation:
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

        if task is None:
            task = self.sample_task()
        self.load_task(task)

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
            if action.action_type == "reason":
                output, reason_reward = self._add_reasoning_step(action.value)
                reward += reason_reward
                info["output"] = output

            elif action.action_type == "calculate":
                result, calc_reward = self._calculate(action.value)
                reward += calc_reward
                info["result"] = result

            elif action.action_type == "verify":
                verified, verify_reward = self._verify_current_answer()
                reward += verify_reward
                info["verified"] = verified

            elif action.action_type == "submit":
                correct, submit_reward = self._submit_answer(action.value)
                reward += submit_reward
                self._done = True
                info["correct"] = correct

            else:
                reward -= 0.1
                info["error"] = f"Unknown action type: {action.action_type}"

        except Exception as e:
            reward -= 0.5
            info["error"] = str(e)

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

    def _add_reasoning_step(self, step: str) -> Tuple[str, float]:
        """Add a reasoning step to the solution.

        Args:
            step: The reasoning step text

        Returns:
            Tuple of (confirmation message, reward)
        """
        self._state.history.append(self._state.current_solution)
        self._state.steps_taken.append(step)

        # Append to current solution
        if self._state.current_solution:
            self._state.current_solution += f"\n{step}"
        else:
            self._state.current_solution = step

        # Check if step contains a potential answer
        reward = 0.02  # Small reward for making progress

        return f"Reasoning step added: {step[:50]}...", reward

    def _calculate(self, expression: str) -> Tuple[str, float]:
        """Use calculator to evaluate an expression.

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            Tuple of (result, reward)
        """
        if not self._allow_calculator:
            return "Calculator not allowed.", -0.1

        self._state.calculator_uses += 1

        try:
            # Safe evaluation using sympy
            result = sympy.sympify(expression)
            result_eval = float(result.evalf())

            # Format result
            if result_eval == int(result_eval):
                result_str = str(int(result_eval))
            else:
                result_str = f"{result_eval:.{self._calculator_precision}g}"

            # Add calculation to solution
            calc_step = f"Calculate: {expression} = {result_str}"
            self._state.steps_taken.append(calc_step)
            if self._state.current_solution:
                self._state.current_solution += f"\n{calc_step}"
            else:
                self._state.current_solution = calc_step

            return result_str, 0.02

        except Exception as e:
            return f"Calculation error: {str(e)}", -0.05

    def _verify_current_answer(self) -> Tuple[bool, float]:
        """Verify the current extracted answer.

        Returns:
            Tuple of (is_correct, reward)
        """
        if self._current_task is None:
            return False, -0.1

        extracted = self._extract_answer(self._state.current_solution)
        if extracted is None:
            return False, -0.05

        is_correct = self._compare_answers(extracted, self._current_task.answer)
        reward = 0.2 if is_correct else -0.1

        return is_correct, reward

    def _submit_answer(self, answer: str) -> Tuple[bool, float]:
        """Submit a final answer.

        Args:
            answer: The final answer to submit

        Returns:
            Tuple of (is_correct, reward)
        """
        if self._current_task is None:
            return False, -0.1

        is_correct = self._compare_answers(answer, self._current_task.answer)

        # Calculate reward
        base_reward = self.config.get("success_reward", 1.0)
        if is_correct:
            reward = base_reward
        else:
            reward = self.config.get("failure_reward", -0.5)

        # Bonus for showing work
        if len(self._state.steps_taken) > 0 and is_correct:
            reward += 0.1

        return is_correct, reward

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract the final answer from solution text.

        Args:
            text: Solution text

        Returns:
            Extracted answer or None
        """
        # Common patterns for answers
        patterns = [
            r"(?:therefore|thus|hence|answer is|final answer|boxed)\s*[:\s]\s*([$\d\.\-]+)",
            r"\$\s*([$\d\.\-]+)\s*\$",
            r"####\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _compare_answers(self, predicted: str, expected: str) -> bool:
        """Compare predicted answer with expected answer.

        Args:
            predicted: Predicted answer
            expected: Expected answer

        Returns:
            True if answers match, False otherwise
        """
        # Normalize answers
        def normalize(s: str) -> str:
            # Remove common separators
            s = s.replace(",", "").replace("$", "").replace("%", "")
            # Try to convert to number
            try:
                num = float(s)
                if num == int(num):
                    return str(int(num))
                return f"{num:.6g}"
            except ValueError:
                return s.lower().strip()

        pred_norm = normalize(predicted)
        exp_norm = normalize(expected)

        # Direct comparison
        if pred_norm == exp_norm:
            return True

        # Try sympy comparison for symbolic expressions
        try:
            pred_sym = sympy.sympify(predicted)
            exp_sym = sympy.sympify(expected)
            return sympy.simplify(pred_sym - exp_sym) == 0
        except Exception:
            pass

        # Fuzzy numeric comparison
        try:
            pred_num = float(pred_norm)
            exp_num = float(exp_norm)
            return abs(pred_num - exp_num) < 1e-6
        except ValueError:
            pass

        return False

    def get_observation(self) -> Observation:
        """Get the current observation.

        Returns:
            Current observation
        """
        text = self.get_task_description()

        if self._state.current_solution:
            text += f"\n\nCurrent solution:\n{self._state.current_solution}"

        text += f"\n\nSteps taken: {len(self._state.steps_taken)}"
        text += f"\nCalculator uses: {self._state.calculator_uses}"

        return Observation(
            text=text,
            state={
                "solution": self._state.current_solution,
                "steps": self._state.steps_taken,
                "step": self._current_step,
            },
            metadata={
                "steps_count": len(self._state.steps_taken),
                "calculator_uses": self._state.calculator_uses,
                "history_length": len(self._state.history),
            },
        )

    def get_action_space(self) -> ActionSpace:
        """Get the action space.

        Returns:
            Action space for math environment
        """
        action_types = ["reason", "submit"]
        if self._allow_calculator:
            action_types.append("calculate")
        action_types.append("verify")

        return ActionSpace(
            action_types=action_types,
            value_space="text",
            constraints={"max_step_length": 500},
        )

    def get_observation_space(self) -> ObservationSpace:
        """Get the observation space.

        Returns:
            Observation space for math environment
        """
        return ObservationSpace(
            text_max_length=4096,
            metadata_keys=["steps_count", "calculator_uses", "history_length"],
        )

    def compute_reward(self, **kwargs) -> float:
        """Compute reward for current state.

        Args:
            **kwargs: Reward computation arguments

        Returns:
            Computed reward
        """
        reward = 0.0

        # Reward for progress (more steps = more progress)
        reward += len(self._state.steps_taken) * 0.01

        # Check if current solution contains correct answer
        if self._state.current_solution:
            extracted = self._extract_answer(self._state.current_solution)
            if extracted and self._current_task:
                if self._compare_answers(extracted, self._current_task.answer):
                    reward += 0.3

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
            "topic": self._current_task.topic if self._current_task else None,
            "steps_taken": len(self._state.steps_taken),
            "calculator_uses": self._state.calculator_uses,
        }

    def is_valid_action(self, action: Action) -> bool:
        """Check if an action is valid.

        Args:
            action: Action to validate

        Returns:
            True if valid, False otherwise
        """
        if action.action_type not in ["reason", "calculate", "verify", "submit"]:
            return False

        if not action.value:
            return False

        if action.action_type == "calculate" and not self._allow_calculator:
            return False

        return True

    def get_task_description(self) -> str:
        """Get task description.

        Returns:
            Task description string
        """
        if self._current_task is None:
            return "No task loaded."

        return f"""Problem ({self._current_task.topic}):

{self._current_task.problem}"""

    def close(self) -> None:
        """Clean up the environment."""
        self._current_task = None
        self._state = MathState()
