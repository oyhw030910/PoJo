"""Metrics Module.

Implements evaluation metrics for RL-LLM Agent performance.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode.

    Attributes:
        episode_id: Episode identifier
        reward: Total episode reward
        length: Episode length
        success: Whether episode was successful
        steps: List of step metrics
    """
    episode_id: int
    reward: float
    length: int
    success: bool
    steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and computes evaluation metrics.

    Tracks metrics across episodes and provides aggregation.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self._episodes: List[EpisodeMetrics] = []
        self._current_episode: int = 0
        self._step_metrics: List[Dict[str, Any]] = []

        # Running statistics
        self._total_reward = 0.0
        self._total_episodes = 0
        self._successful_episodes = 0
        self._total_steps = 0

    def start_episode(self, episode_id: Optional[int] = None) -> int:
        """Start a new episode.

        Args:
            episode_id: Optional episode ID

        Returns:
            Episode ID
        """
        if episode_id is None:
            episode_id = self._current_episode

        self._current_episode = episode_id
        self._step_metrics = []

        return episode_id

    def record_step(
        self,
        reward: float,
        observation: Optional[str] = None,
        action: Optional[str] = None,
        **kwargs
    ) -> None:
        """Record a step within an episode.

        Args:
            reward: Reward received
            observation: Environment observation
            action: Agent action
            **kwargs: Additional metrics
        """
        step_data = {
            "reward": reward,
            "observation": observation,
            "action": action,
            **kwargs,
        }
        self._step_metrics.append(step_data)

    def end_episode(
        self,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EpisodeMetrics:
        """End the current episode.

        Args:
            success: Whether episode was successful
            metadata: Optional metadata

        Returns:
            Episode metrics
        """
        total_reward = sum(s.get("reward", 0) for s in self._step_metrics)
        length = len(self._step_metrics)

        episode = EpisodeMetrics(
            episode_id=self._current_episode,
            reward=total_reward,
            length=length,
            success=success,
            steps=self._step_metrics.copy(),
            metadata=metadata or {},
        )

        self._episodes.append(episode)

        # Update running statistics
        self._total_reward += total_reward
        self._total_episodes += 1
        self._total_steps += length
        if success:
            self._successful_episodes += 1

        return episode

    def get_episode_metrics(self, episode_id: int) -> Optional[EpisodeMetrics]:
        """Get metrics for a specific episode.

        Args:
            episode_id: Episode ID

        Returns:
            Episode metrics or None
        """
        for ep in self._episodes:
            if ep.episode_id == episode_id:
                return ep
        return None

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Get aggregate metrics across all episodes.

        Returns:
            Dictionary of aggregate metrics
        """
        if not self._episodes:
            return {
                "episodes": 0,
                "success_rate": 0.0,
                "avg_reward": 0.0,
                "avg_length": 0.0,
            }

        rewards = [ep.reward for ep in self._episodes]
        lengths = [ep.length for ep in self._episodes]
        successes = [1 if ep.success else 0 for ep in self._episodes]

        return {
            "episodes": len(self._episodes),
            "success_rate": np.mean(successes),
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "avg_length": np.mean(lengths),
            "avg_steps": self._total_steps / self._total_episodes,
            "total_reward": self._total_reward,
        }

    def get_reward_curve(self) -> List[float]:
        """Get reward curve (reward per episode).

        Returns:
            List of rewards per episode
        """
        return [ep.reward for ep in self._episodes]

    def get_success_curve(self) -> List[int]:
        """Get success per episode.

        Returns:
            List of success flags
        """
        return [1 if ep.success else 0 for ep in self._episodes]

    def get_windowed_metrics(self, window_size: int = 10) -> Dict[str, List[float]]:
        """Get windowed metrics for trend analysis.

        Args:
            window_size: Window size for averaging

        Returns:
            Dictionary of windowed metrics
        """
        if len(self._episodes) < window_size:
            return self.get_aggregate_metrics()

        rewards = [ep.reward for ep in self._episodes]
        successes = [1 if ep.success else 0 for ep in self._episodes]

        windowed_rewards = []
        windowed_success_rates = []

        for i in range(len(rewards) - window_size + 1):
            window = rewards[i:i + window_size]
            windowed_rewards.append(np.mean(window))
            windowed_success_rates.append(np.mean(successes[i:i + window_size]))

        return {
            "windowed_rewards": windowed_rewards,
            "windowed_success_rates": windowed_success_rates,
        }

    def get_progress_metrics(self) -> Dict[str, Any]:
        """Get metrics showing training progress.

        Returns:
            Dictionary of progress metrics
        """
        if len(self._episodes) < 2:
            return {"progress": "insufficient_data"}

        # Compare first half vs second half
        mid = len(self._episodes) // 2
        first_half = self._episodes[:mid]
        second_half = self._episodes[mid:]

        first_reward = np.mean([ep.reward for ep in first_half])
        second_reward = np.mean([ep.reward for ep in second_half])

        first_success = np.mean([1 if ep.success else 0 for ep in first_half])
        second_success = np.mean([1 if ep.success else 0 for ep in second_half])

        return {
            "reward_improvement": (second_reward - first_reward) / (abs(first_reward) + 1e-6),
            "success_improvement": second_success - first_success,
            "first_half_reward": first_reward,
            "second_half_reward": second_reward,
            "first_half_success": first_success,
            "second_half_success": second_success,
        }

    def clear(self) -> None:
        """Clear all metrics."""
        self._episodes = []
        self._total_reward = 0.0
        self._total_episodes = 0
        self._successful_episodes = 0
        self._total_steps = 0

    def export_to_dict(self) -> Dict[str, Any]:
        """Export metrics to dictionary.

        Returns:
            Dictionary with all metrics
        """
        return {
            "aggregate": self.get_aggregate_metrics(),
            "progress": self.get_progress_metrics(),
            "reward_curve": self.get_reward_curve(),
            "success_curve": self.get_success_curve(),
            "episodes": [
                {
                    "episode_id": ep.episode_id,
                    "reward": ep.reward,
                    "length": ep.length,
                    "success": ep.success,
                }
                for ep in self._episodes
            ],
        }


class TaskSpecificMetrics:
    """Task-specific metrics for different environments."""

    def __init__(self):
        """Initialize task-specific metrics."""
        self._code_metrics: Dict[str, Any] = {
            "pass_rate": [],
            "test_cases_passed": [],
        }
        self._math_metrics: Dict[str, Any] = {
            "accuracy": [],
            "steps_used": [],
        }
        self._gui_metrics: Dict[str, Any] = {
            "success_rate": [],
            "actions_taken": [],
        }

    def record_code_result(
        self,
        pass_rate: float,
        tests_passed: int,
        tests_total: int
    ) -> None:
        """Record code generation result.

        Args:
            pass_rate: Test pass rate
            tests_passed: Number of tests passed
            tests_total: Total number of tests
        """
        self._code_metrics["pass_rate"].append(pass_rate)
        self._code_metrics["test_cases_passed"].append(tests_passed / tests_total)

    def record_math_result(
        self,
        correct: bool,
        steps_used: int
    ) -> None:
        """Record math solving result.

        Args:
            correct: Whether answer was correct
            steps_used: Number of steps used
        """
        self._math_metrics["accuracy"].append(1 if correct else 0)
        self._math_metrics["steps_used"].append(steps_used)

    def record_gui_result(
        self,
        success: bool,
        actions_taken: int
    ) -> None:
        """Record GUI navigation result.

        Args:
            success: Whether task was completed
            actions_taken: Number of actions taken
        """
        self._gui_metrics["success_rate"].append(1 if success else 0)
        self._gui_metrics["actions_taken"].append(actions_taken)

    def get_code_metrics(self) -> Dict[str, float]:
        """Get code generation metrics.

        Returns:
            Dictionary of metrics
        """
        if not self._code_metrics["pass_rate"]:
            return {"avg_pass_rate": 0.0}

        return {
            "avg_pass_rate": np.mean(self._code_metrics["pass_rate"]),
            "avg_test_completion": np.mean(self._code_metrics["test_cases_passed"]),
        }

    def get_math_metrics(self) -> Dict[str, float]:
        """Get math solving metrics.

        Returns:
            Dictionary of metrics
        """
        if not self._math_metrics["accuracy"]:
            return {"avg_accuracy": 0.0, "avg_steps": 0.0}

        return {
            "avg_accuracy": np.mean(self._math_metrics["accuracy"]),
            "avg_steps": np.mean(self._math_metrics["steps_used"]),
            "efficiency": np.mean(self._math_metrics["accuracy"]) / (np.mean(self._math_metrics["steps_used"]) + 1),
        }

    def get_gui_metrics(self) -> Dict[str, float]:
        """Get GUI navigation metrics.

        Returns:
            Dictionary of metrics
        """
        if not self._gui_metrics["success_rate"]:
            return {"avg_success_rate": 0.0, "avg_actions": 0.0}

        return {
            "avg_success_rate": np.mean(self._gui_metrics["success_rate"]),
            "avg_actions": np.mean(self._gui_metrics["actions_taken"]),
        }

    def clear(self) -> None:
        """Clear all task-specific metrics."""
        for key in self._code_metrics:
            self._code_metrics[key] = []
        for key in self._math_metrics:
            self._math_metrics[key] = []
        for key in self._gui_metrics:
            self._gui_metrics[key] = []


def compute_sample_efficiency(
    rewards: List[float],
    num_steps: int,
    target_reward: float
) -> float:
    """Compute sample efficiency metric.

    Args:
        rewards: List of rewards per episode
        num_steps: Total environment steps
        target_reward: Target reward level

    Returns:
        Sample efficiency score
    """
    # Find how many episodes to reach target reward
    cumulative = 0
    episodes_to_target = 0

    for i, reward in enumerate(rewards):
        cumulative += reward
        if cumulative >= target_reward:
            episodes_to_target = i + 1
            break

    if episodes_to_target == 0:
        return 0.0

    # Efficiency = target / (steps taken)
    return target_reward / (num_steps / len(rewards) * episodes_to_target)


def compute_success_at_k(
    successes: List[int],
    k: int
) -> float:
    """Compute success rate at k episodes.

    Args:
        successes: List of success flags
        k: Number of episodes to consider

    Returns:
        Success rate at k
    """
    if len(successes) < k:
        return np.mean(successes) if successes else 0.0

    return np.mean(successes[:k])


def compute_area_under_curve(
    values: List[float],
    normalize: bool = True
) -> float:
    """Compute area under the learning curve.

    Args:
        values: Values over time
        normalize: Whether to normalize to [0, 1]

    Returns:
        Area under curve
    """
    auc = np.trapz(values)

    if normalize and len(values) > 0:
        max_auc = len(values) * max(values) if max(values) > 0 else 1
        auc = auc / max_auc

    return auc
