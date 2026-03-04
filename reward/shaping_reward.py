"""Shaping Reward Module.

Provides reward shaping functions to guide agent learning.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from .base_reward import BaseReward, RewardInfo


@dataclass
class ShapingConfig:
    """Configuration for shaping rewards.

    Attributes:
        step_penalty: Penalty per step (encourages efficiency)
        format_bonus: Bonus for correct output format
        intermediate_bonus: Bonus for correct intermediate steps
        progress_bonus: Bonus for making progress
    """
    step_penalty: float = -0.01
    format_bonus: float = 0.1
    intermediate_bonus: float = 0.1
    progress_bonus: float = 0.2


class ShapingReward(BaseReward):
    """Reward shaping for guiding agent learning.

    This reward function provides additional signals to guide the agent
    during training. It includes:
    - Step penalty (encourage efficiency)
    - Format bonuses (correct output structure)
    - Intermediate step rewards
    - Progress-based rewards

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self._step_penalty = self.config.get("step_penalty", -0.01)
        self._format_bonus = self.config.get("format_bonus", 0.1)
        self._intermediate_bonus = self.config.get("intermediate_bonus", 0.1)
        self._progress_bonus = self.config.get("progress_bonus", 0.2)

        # State tracking
        self._previous_state: Optional[Any] = None
        self._achieved_milestones: List[str] = []

    def compute(self, **kwargs) -> float:
        """Compute shaping reward.

        Args:
            **kwargs: Various shaping reward inputs:
                - current_step: Current step number
                - max_steps: Maximum allowed steps
                - format_valid: Whether output format is valid
                - intermediate_correct: Whether intermediate steps are correct
                - progress: Progress metric (0-1)
                - state: Current state for potential-based shaping

        Returns:
            Computed shaping reward
        """
        total_reward = 0.0

        # Step penalty (always applied)
        total_reward += self._step_penalty

        # Format bonus
        if kwargs.get("format_valid", False):
            total_reward += self._format_bonus

        # Intermediate step bonus
        if kwargs.get("intermediate_correct", False):
            total_reward += self._intermediate_bonus

        # Progress-based reward
        progress = kwargs.get("progress", None)
        if progress is not None:
            total_reward += self._compute_progress_reward(progress)

        # Potential-based shaping (if state provided)
        state = kwargs.get("state", None)
        if state is not None and self._previous_state is not None:
            total_reward += self._compute_potential_shaping(state)

        self._previous_state = state
        return total_reward

    def compute_with_info(self, **kwargs) -> RewardInfo:
        """Compute shaping reward with breakdown.

        Args:
            **kwargs: Arguments for reward computation

        Returns:
            RewardInfo with component breakdown
        """
        components = {}
        metadata = {}

        # Step penalty
        components["step_penalty"] = self._step_penalty
        total = self._step_penalty

        # Format bonus
        if kwargs.get("format_valid", False):
            components["format_bonus"] = self._format_bonus
            total += self._format_bonus
            metadata["format_valid"] = True

        # Intermediate bonus
        if kwargs.get("intermediate_correct", False):
            components["intermediate_bonus"] = self._intermediate_bonus
            total += self._intermediate_bonus
            metadata["intermediate_correct"] = True

        # Progress reward
        progress = kwargs.get("progress", None)
        if progress is not None:
            progress_reward = self._compute_progress_reward(progress)
            components["progress_reward"] = progress_reward
            total += progress_reward
            metadata["progress"] = progress

        # Potential-based shaping
        state = kwargs.get("state", None)
        if state is not None and self._previous_state is not None:
            potential_reward = self._compute_potential_shaping(state)
            components["potential_shaping"] = potential_reward
            total += potential_reward

        return RewardInfo(
            reward=total,
            components=components,
            metadata=metadata,
        )

    def _compute_progress_reward(self, progress: float) -> float:
        """Compute progress-based reward.

        Args:
            progress: Progress value (0-1)

        Returns:
            Progress reward
        """
        return self._progress_bonus * progress

    def _compute_potential_shaping(self, state: Any) -> float:
        """Compute potential-based reward shaping.

        Uses the formula: F(s, s') = gamma * Phi(s') - Phi(s)

        Args:
            state: Current state

        Returns:
            Potential-based shaping reward
        """
        # Default potential function: negative distance to goal
        # Override in subclasses for specific environments
        gamma = self.config.get("gamma", 0.99)

        potential_current = self._potential(self._previous_state)
        potential_next = self._potential(state)

        return gamma * potential_next - potential_current

    def _potential(self, state: Any) -> float:
        """Potential function for state.

        Args:
            state: State to evaluate

        Returns:
            Potential value (higher = closer to goal)
        """
        # Default: return 0 (no potential shaping)
        # Override in subclasses
        return 0.0

    def set_milestone_reward(self, milestone: str, achieved: bool) -> float:
        """Set and compute milestone reward.

        Args:
            milestone: Milestone identifier
            achieved: Whether milestone was achieved

        Returns:
            Milestone reward
        """
        if achieved and milestone not in self._achieved_milestones:
            self._achieved_milestones.append(milestone)
            return self._intermediate_bonus
        return 0.0

    def reset(self) -> None:
        """Reset shaping reward state."""
        self._previous_state = None
        self._achieved_milestones = []


class FormatReward(BaseReward):
    """Reward for correct output format.

    Provides rewards when the agent produces output in the correct format.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self._format_checks: Dict[str, Callable] = {}
        self._format_rewards: Dict[str, float] = {}

    def add_format_check(
        self,
        format_name: str,
        check_fn: Callable[[str], bool],
        reward: float = 0.1
    ) -> None:
        """Add a format check.

        Args:
            format_name: Name of the format
            check_fn: Function that checks if output matches format
            reward: Reward for matching this format
        """
        self._format_checks[format_name] = check_fn
        self._format_rewards[format_name] = reward

    def compute(self, **kwargs) -> float:
        """Compute format reward.

        Args:
            **kwargs: Required:
                - output: Agent output string

        Returns:
            Format reward
        """
        output = kwargs.get("output", "")
        if not output:
            return 0.0

        total_reward = 0.0

        for format_name, check_fn in self._format_checks.items():
            try:
                if check_fn(output):
                    total_reward += self._format_rewards[format_name]
            except Exception:
                pass

        return total_reward

    def compute_with_info(self, **kwargs) -> RewardInfo:
        """Compute format reward with breakdown.

        Args:
            **kwargs: Arguments for reward computation

        Returns:
            RewardInfo with per-format breakdown
        """
        output = kwargs.get("output", "")
        components = {}
        total = 0.0
        metadata = {"matched_formats": []}

        for format_name, check_fn in self._format_checks.items():
            try:
                if check_fn(output):
                    components[format_name] = self._format_rewards[format_name]
                    total += self._format_rewards[format_name]
                    metadata["matched_formats"].append(format_name)
            except Exception:
                pass

        return RewardInfo(
            reward=total,
            components=components,
            metadata=metadata,
        )


class EfficiencyReward(BaseReward):
    """Reward for efficient task completion.

    Provides rewards based on how efficiently the agent completes tasks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self._optimal_steps = self.config.get("optimal_steps", 1)
        self._max_steps = self.config.get("max_steps", 100)
        self._base_reward = self.config.get("base_reward", 1.0)

    def compute(self, **kwargs) -> float:
        """Compute efficiency reward.

        Args:
            **kwargs: Required:
                - steps_taken: Number of steps taken
                - success: Whether task was successful

        Returns:
            Efficiency reward
        """
        steps_taken = kwargs.get("steps_taken", 1)
        success = kwargs.get("success", True)

        if not success:
            return self.config.get("failure_reward", -0.5)

        # Efficiency = optimal / actual
        efficiency = min(1.0, self._optimal_steps / max(1, steps_taken))

        # Scale reward by efficiency
        return self._base_reward * efficiency

    def compute_with_info(self, **kwargs) -> RewardInfo:
        """Compute efficiency reward with breakdown.

        Args:
            **kwargs: Arguments for reward computation

        Returns:
            RewardInfo with efficiency breakdown
        """
        steps_taken = kwargs.get("steps_taken", 1)
        success = kwargs.get("success", True)

        if not success:
            return RewardInfo(
                reward=self.config.get("failure_reward", -0.5),
                components={"failure": self.config.get("failure_reward", -0.5)},
                metadata={"success": False},
            )

        efficiency = min(1.0, self._optimal_steps / max(1, steps_taken))
        reward = self._base_reward * efficiency

        return RewardInfo(
            reward=reward,
            components={
                "base": self._base_reward,
                "efficiency_multiplier": efficiency,
            },
            metadata={
                "steps_taken": steps_taken,
                "optimal_steps": self._optimal_steps,
                "efficiency": efficiency,
            },
        )

    def set_optimal_steps(self, steps: int) -> None:
        """Set the optimal number of steps.

        Args:
            steps: Optimal step count
        """
        self._optimal_steps = steps
