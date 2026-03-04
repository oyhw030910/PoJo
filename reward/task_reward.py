"""Task Completion Reward Module.

Provides reward functions based on task completion status.
"""

from typing import Any, Dict, List, Optional
from .base_reward import BaseReward, RewardInfo


class TaskReward(BaseReward):
    """Reward based on task completion.

    This reward function provides rewards based on whether the agent
    successfully completes the task. It supports:
    - Binary success/failure rewards
    - Partial completion rewards
    - Multi-task reward aggregation

    Args:
        config: Configuration dictionary with optional keys:
            - success_reward: Reward for task success (default: 1.0)
            - failure_reward: Reward for task failure (default: -1.0)
            - partial_rewards: Dictionary of partial completion rewards
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self._success_reward = self.config.get("success_reward", 1.0)
        self._failure_reward = self.config.get("failure_reward", -1.0)
        self._partial_rewards = self.config.get("partial_rewards", {})

    def compute(self, **kwargs) -> float:
        """Compute task completion reward.

        Args:
            **kwargs: Required keywords:
                - success: Boolean indicating task success
                - partial: Optional float for partial completion (0-1)
                - metrics: Optional dict with task-specific metrics

        Returns:
            Computed reward value
        """
        success = kwargs.get("success")
        partial = kwargs.get("partial", None)
        metrics = kwargs.get("metrics", {})

        if success is True:
            return self._success_reward
        elif success is False:
            return self._failure_reward
        elif partial is not None:
            # Partial completion
            return self._failure_reward + (self._success_reward - self._failure_reward) * partial
        else:
            # Try to compute from metrics
            return self._compute_from_metrics(metrics)

    def compute_with_info(self, **kwargs) -> RewardInfo:
        """Compute reward with detailed breakdown.

        Args:
            **kwargs: Arguments for reward computation

        Returns:
            RewardInfo with detailed breakdown
        """
        success = kwargs.get("success")
        partial = kwargs.get("partial", None)
        metrics = kwargs.get("metrics", {})

        components = {}
        metadata = {"success": success, "partial": partial}

        if success is True:
            components["success"] = self._success_reward
            total = self._success_reward
        elif success is False:
            components["failure"] = self._failure_reward
            total = self._failure_reward
        elif partial is not None:
            base = self._failure_reward
            bonus = (self._success_reward - self._failure_reward) * partial
            components["base"] = base
            components["partial_bonus"] = bonus
            total = base + bonus
            metadata["partial_ratio"] = partial
        else:
            total = self._compute_from_metrics(metrics)
            components["metrics"] = total

        return RewardInfo(
            reward=total,
            components=components,
            metadata=metadata,
        )

    def _compute_from_metrics(self, metrics: Dict[str, Any]) -> float:
        """Compute reward from task metrics.

        Args:
            metrics: Dictionary of task metrics

        Returns:
            Computed reward
        """
        # Default: check for pass_rate or similar metric
        if "pass_rate" in metrics:
            pass_rate = metrics["pass_rate"]
            return self._failure_reward + (self._success_reward - self._failure_reward) * pass_rate

        if "accuracy" in metrics:
            accuracy = metrics["accuracy"]
            return self._failure_reward + (self._success_reward - self._failure_reward) * accuracy

        # Check for test case results
        if "tests_passed" in metrics and "tests_total" in metrics:
            passed = metrics["tests_passed"]
            total = metrics["tests_total"]
            if total > 0:
                pass_rate = passed / total
                return self._failure_reward + (self._success_reward - self._failure_reward) * pass_rate

        # No metrics matched
        return 0.0

    def set_success_reward(self, value: float) -> None:
        """Set the success reward value.

        Args:
            value: New success reward value
        """
        self._success_reward = value

    def set_failure_reward(self, value: float) -> None:
        """Set the failure reward value.

        Args:
            value: New failure reward value
        """
        self._failure_reward = value


class MultiTaskReward(BaseReward):
    """Reward for multiple tasks.

    Aggregates rewards from multiple tasks or subtasks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self._task_rewards: Dict[str, TaskReward] = {}
        self._weights: Dict[str, float] = {}

    def add_task(self, task_id: str, reward_fn: Optional[TaskReward] = None, weight: float = 1.0) -> None:
        """Add a task to the multi-task reward.

        Args:
            task_id: Unique task identifier
            reward_fn: Optional TaskReward instance (creates default if None)
            weight: Weight for this task's reward
        """
        if reward_fn is None:
            reward_fn = TaskReward(self.config)
        self._task_rewards[task_id] = reward_fn
        self._weights[task_id] = weight

    def remove_task(self, task_id: str) -> None:
        """Remove a task.

        Args:
            task_id: Task identifier to remove
        """
        if task_id in self._task_rewards:
            del self._task_rewards[task_id]
            del self._weights[task_id]

    def compute(self, **kwargs) -> float:
        """Compute aggregated multi-task reward.

        Args:
            **kwargs: Dictionary with task_id -> task_kwargs mapping

        Returns:
            Weighted sum of task rewards
        """
        total_reward = 0.0
        total_weight = 0.0

        for task_id, task_kwargs in kwargs.items():
            if task_id in self._task_rewards:
                weight = self._weights[task_id]
                reward_fn = self._task_rewards[task_id]

                if isinstance(task_kwargs, dict):
                    task_reward = reward_fn.compute(**task_kwargs)
                else:
                    task_reward = task_kwargs  # Already computed reward

                total_reward += weight * task_reward
                total_weight += weight

        if total_weight > 0:
            return total_reward / total_weight
        return 0.0

    def compute_with_info(self, **kwargs) -> RewardInfo:
        """Compute multi-task reward with breakdown.

        Args:
            **kwargs: Task rewards mapping

        Returns:
            RewardInfo with per-task breakdown
        """
        components = {}
        total_reward = 0.0
        total_weight = 0.0

        for task_id, task_kwargs in kwargs.items():
            if task_id in self._task_rewards:
                weight = self._weights[task_id]
                reward_fn = self._task_rewards[task_id]

                if isinstance(task_kwargs, dict):
                    info = reward_fn.compute_with_info(**task_kwargs)
                else:
                    info = RewardInfo(reward=task_kwargs, components={}, metadata={})

                components[f"{task_id}"] = weight * info.reward
                total_reward += weight * info.reward
                total_weight += weight

        if total_weight > 0:
            total_reward /= total_weight

        return RewardInfo(
            reward=total_reward,
            components=components,
            metadata={"weights": self._weights.copy()},
        )

    def reset(self) -> None:
        """Reset all task rewards."""
        for reward_fn in self._task_rewards.values():
            reward_fn.reset()
