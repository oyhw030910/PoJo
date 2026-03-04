"""Evaluator Module.

Implements evaluation procedures for RL-LLM Agent.
"""

import os
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import torch

from .metrics import MetricsCollector, TaskSpecificMetrics, EpisodeMetrics


@dataclass
class EvaluatorConfig:
    """Configuration for evaluation.

    Attributes:
        num_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic actions
        save_results: Whether to save results
        results_dir: Directory for saving results
        eval_interval: Steps between evaluations
        timeout: Episode timeout in seconds
        log_to_file: Whether to log to file
    """
    num_episodes: int = 100
    deterministic: bool = True
    save_results: bool = True
    results_dir: str = "./evaluation_results"
    eval_interval: int = 1000
    timeout: int = 300
    log_to_file: bool = True


@dataclass
class EvaluationResult:
    """Result of an evaluation run.

    Attributes:
        timestamp: Evaluation timestamp
        num_episodes: Number of episodes
        success_rate: Success rate
        avg_reward: Average reward
        avg_episode_length: Average episode length
        metrics: Full metrics dictionary
        task_metrics: Task-specific metrics
    """
    timestamp: str
    num_episodes: int
    success_rate: float
    avg_reward: float
    avg_episode_length: float
    metrics: Dict[str, Any]
    task_metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class Evaluator:
    """Main evaluator for RL-LLM Agent.

    Provides comprehensive evaluation capabilities including:
    - Multi-episode evaluation
    - Task-specific metrics
    - Result saving and logging
    - Comparison with baselines

    Usage:
        config = EvaluatorConfig(num_episodes=100)
        evaluator = Evaluator(policy, env, config)
        result = evaluator.evaluate()
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        env: Any,
        config: Optional[EvaluatorConfig] = None,
        device: Union[str, torch.device] = "cpu",
        obs_to_tensor: Optional[Any] = None,
        action_to_tensor: Optional[Any] = None,
    ):
        """Initialize evaluator.

        Args:
            policy: Policy model to evaluate
            env: Environment or environment factory
            config: Evaluator configuration
            device: Device for evaluation
            obs_to_tensor: Function to convert observations to tensors
            action_to_tensor: Function to convert actions to tensors
        """
        self.policy = policy
        self.env = env
        self.config = config or EvaluatorConfig()
        self.device = torch.device(device)

        # Conversion functions
        self.obs_to_tensor = obs_to_tensor
        self.action_to_tensor = action_to_tensor

        # Metrics collectors
        self.metrics_collector = MetricsCollector()
        self.task_metrics = TaskSpecificMetrics()

        # Results storage
        self._results: List[EvaluationResult] = []
        self._episode_details: List[Dict[str, Any]] = []

        # Create results directory
        if self.config.save_results:
            os.makedirs(self.config.results_dir, exist_ok=True)

    def evaluate(
        self,
        num_episodes: Optional[int] = None,
        deterministic: Optional[bool] = None,
        verbose: bool = True
    ) -> EvaluationResult:
        """Run evaluation.

        Args:
            num_episodes: Number of episodes (overrides config)
            deterministic: Whether to use deterministic actions
            verbose: Whether to print progress

        Returns:
            Evaluation result
        """
        num_episodes = num_episodes or self.config.num_episodes
        deterministic = deterministic if deterministic is not None else self.config.deterministic

        # Set policy to eval mode
        self.policy.eval()

        # Reset metrics
        self.metrics_collector.clear()
        self.task_metrics.clear()

        # Run episodes
        with torch.no_grad():
            for episode_idx in range(num_episodes):
                episode_result = self._run_episode(
                    episode_idx=episode_idx,
                    deterministic=deterministic,
                )
                self._episode_details.append(episode_result)

                if verbose and (episode_idx + 1) % 10 == 0:
                    current_metrics = self.metrics_collector.get_aggregate_metrics()
                    print(f"Episode {episode_idx + 1}/{num_episodes}: "
                          f"Success Rate: {current_metrics['success_rate']:.2%}, "
                          f"Avg Reward: {current_metrics['avg_reward']:.3f}")

        # Compile results
        aggregate_metrics = self.metrics_collector.get_aggregate_metrics()
        task_specific = self._get_task_specific_metrics()

        result = EvaluationResult(
            timestamp=datetime.now().isoformat(),
            num_episodes=num_episodes,
            success_rate=aggregate_metrics["success_rate"],
            avg_reward=aggregate_metrics["avg_reward"],
            avg_episode_length=aggregate_metrics["avg_length"],
            metrics=aggregate_metrics,
            task_metrics=task_specific,
        )

        self._results.append(result)

        # Save results
        if self.config.save_results:
            self._save_results(result)

        if verbose:
            self._print_summary(result)

        return result

    def _run_episode(
        self,
        episode_idx: int,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """Run a single evaluation episode.

        Args:
            episode_idx: Episode index
            deterministic: Whether to use deterministic actions

        Returns:
            Episode result dictionary
        """
        # Reset environment
        obs = self.env.reset()
        done = False
        truncated = False

        # Start metrics recording
        self.metrics_collector.start_episode(episode_idx)

        episode_start = time.time()
        step_count = 0

        while not (done or truncated):
            # Convert observation
            if self.obs_to_tensor:
                obs_tensor = self.obs_to_tensor(obs)
            else:
                obs_tensor = self._default_obs_to_tensor(obs)

            # Get action
            with torch.no_grad():
                if deterministic:
                    action = self.policy.get_action(
                        obs_tensor,
                        deterministic=True,
                    )
                else:
                    action, log_prob = self.policy.get_action_with_log_prob(obs_tensor)

            # Convert action
            action_np = self._convert_action(action)

            # Take step
            result = self.env.step(action_np)
            obs = result.observation
            reward = result.reward
            done = result.done
            truncated = result.truncated
            info = result.info

            # Record step
            self.metrics_collector.record_step(
                reward=reward,
                observation=obs.text if hasattr(obs, 'text') else str(obs),
                action=str(action_np),
            )

            step_count += 1

            # Check timeout
            if time.time() - episode_start > self.config.timeout:
                truncated = True
                break

        # Determine success
        success = info.get("success", info.get("correct", info.get("goal_reached", False)))
        if isinstance(success, torch.Tensor):
            success = success.item()

        # End metrics recording
        self.metrics_collector.end_episode(
            success=bool(success),
            metadata={
                "steps": step_count,
                "duration": time.time() - episode_start,
            }
        )

        # Record task-specific metrics
        self._record_task_metrics(info)

        return {
            "episode_id": episode_idx,
            "reward": sum(s.get("reward", 0) for s in self.metrics_collector._step_metrics),
            "length": step_count,
            "success": bool(success),
            "info": info,
        }

    def _default_obs_to_tensor(self, obs: Any) -> torch.Tensor:
        """Default observation to tensor conversion.

        Args:
            obs: Observation

        Returns:
            Tensor representation
        """
        if hasattr(obs, 'text'):
            # Text observation - use policy's LLM
            if hasattr(self.policy, 'llm'):
                inputs = self.policy.llm.tokenizer(
                    obs.text,
                    return_tensors="pt",
                    padding=True,
                )
                return {
                    "input_ids": inputs["input_ids"].to(self.device),
                    "attention_mask": inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).to(self.device),
                }

        # Fallback: simple tensor conversion
        if isinstance(obs, (int, float)):
            return torch.tensor([[obs]], device=self.device)

        return torch.tensor([[0]], device=self.device)

    def _convert_action(self, action: torch.Tensor) -> Any:
        """Convert action tensor to environment action.

        Args:
            action: Action tensor

        Returns:
            Environment action
        """
        if self.action_to_tensor:
            return self.action_to_tensor(action)

        # Default: return as numpy
        if isinstance(action, torch.Tensor):
            return action.cpu().numpy()
        return action

    def _record_task_metrics(self, info: Dict[str, Any]) -> None:
        """Record task-specific metrics.

        Args:
            info: Environment info dictionary
        """
        # Code metrics
        if "test_results" in info:
            results = info["test_results"]
            if results:
                passed = sum(1 for r in results if r.get("passed", False))
                total = len(results)
                self.task_metrics.record_code_result(
                    pass_rate=passed / total,
                    tests_passed=passed,
                    tests_total=total,
                )

        # Math metrics
        if "correct" in info and "steps_taken" in info:
            self.task_metrics.record_math_result(
                correct=info["correct"],
                steps_used=info["steps_taken"],
            )

        # GUI metrics
        if "actions_taken" in info:
            self.task_metrics.record_gui_result(
                success=info.get("goal_reached", False),
                actions_taken=info["actions_taken"],
            )

    def _get_task_specific_metrics(self) -> Dict[str, Any]:
        """Get task-specific metrics.

        Returns:
            Dictionary of task metrics
        """
        return {
            "code": self.task_metrics.get_code_metrics(),
            "math": self.task_metrics.get_math_metrics(),
            "gui": self.task_metrics.get_gui_metrics(),
        }

    def _save_results(self, result: EvaluationResult) -> None:
        """Save evaluation results.

        Args:
            result: Evaluation result
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{timestamp}.json"
        filepath = os.path.join(self.config.results_dir, filename)

        data = {
            "result": {
                "timestamp": result.timestamp,
                "num_episodes": result.num_episodes,
                "success_rate": result.success_rate,
                "avg_reward": result.avg_reward,
                "avg_episode_length": result.avg_episode_length,
                "metrics": result.metrics,
                "task_metrics": result.task_metrics,
            },
            "episodes": self._episode_details,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _print_summary(self, result: EvaluationResult) -> None:
        """Print evaluation summary.

        Args:
            result: Evaluation result
        """
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Timestamp: {result.timestamp}")
        print(f"Episodes: {result.num_episodes}")
        print(f"Success Rate: {result.success_rate:.2%}")
        print(f"Average Reward: {result.avg_reward:.3f}")
        print(f"Average Episode Length: {result.avg_episode_length:.1f}")

        if result.task_metrics:
            print("\nTask-Specific Metrics:")
            for task, metrics in result.task_metrics.items():
                if any(v not in [0, 0.0, {}] for v in metrics.values()):
                    print(f"  {task}:")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            print(f"    {key}: {value:.4f}")
                        else:
                            print(f"    {key}: {value}")

        print("=" * 50)

    def compare_with_baseline(
        self,
        baseline_results: List[float],
        metric: str = "reward"
    ) -> Dict[str, float]:
        """Compare current results with baseline.

        Args:
            baseline_results: Baseline metric values
            metric: Metric to compare

        Returns:
            Comparison statistics
        """
        current_rewards = self.metrics_collector.get_reward_curve()

        if not baseline_results:
            return {"error": "No baseline results provided"}

        baseline_mean = np.mean(baseline_results)
        baseline_std = np.std(baseline_results)
        current_mean = np.mean(current_rewards)
        current_std = np.std(current_rewards)

        # Compute improvement
        improvement = (current_mean - baseline_mean) / (abs(baseline_mean) + 1e-6)

        # Statistical significance (simple t-test approximation)
        n1, n2 = len(baseline_results), len(current_rewards)
        se = np.sqrt(baseline_std**2 / n1 + current_std**2 / n2)
        if se > 0:
            t_stat = (current_mean - baseline_mean) / se
        else:
            t_stat = 0

        return {
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "current_mean": current_mean,
            "current_std": current_std,
            "improvement": improvement,
            "improvement_percent": improvement * 100,
            "t_statistic": t_stat,
            "significant": abs(t_stat) > 2,
        }

    def get_training_curve(
        self,
        window_size: int = 10
    ) -> Dict[str, List[float]]:
        """Get training/learning curve.

        Args:
            window_size: Window size for smoothing

        Returns:
            Dictionary with curve data
        """
        return self.metrics_collector.get_windowed_metrics(window_size)

    def export_results(self, filename: Optional[str] = None) -> str:
        """Export all results to file.

        Args:
            filename: Optional filename

        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"all_results_{timestamp}.json"

        filepath = os.path.join(self.config.results_dir, filename)

        data = {
            "evaluations": [
                {
                    "timestamp": r.timestamp,
                    "success_rate": r.success_rate,
                    "avg_reward": r.avg_reward,
                    "metrics": r.metrics,
                    "task_metrics": r.task_metrics,
                }
                for r in self._results
            ],
            "episode_details": self._episode_details,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    def get_best_episode(self) -> Optional[Dict[str, Any]]:
        """Get the best episode by reward.

        Returns:
            Best episode details or None
        """
        if not self._episode_details:
            return None

        return max(self._episode_details, key=lambda x: x.get("reward", 0))

    def get_worst_episode(self) -> Optional[Dict[str, Any]]:
        """Get the worst episode by reward.

        Returns:
            Worst episode details or None
        """
        if not self._episode_details:
            return None

        return min(self._episode_details, key=lambda x: x.get("reward", 0))
