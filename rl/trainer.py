"""RL Trainer Module.

Main trainer class that orchestrates RL training for LLM Agents.
"""

import os
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .ppo_trainer import PPOTrainer, PPOConfig, PPOStats
from .grpo_trainer import GRPOTrainer, GRPOConfig, GRPOStats
from .replay_buffer import ReplayBuffer, SequenceBuffer, SequenceData, Transition


@dataclass
class TrainingConfig:
    """Configuration for RL training.

    Attributes:
        algorithm: RL algorithm ('ppo' or 'grpo')
        total_steps: Total training steps
        rollout_steps: Steps per rollout
        eval_interval: Steps between evaluations
        save_interval: Steps between checkpoints
        log_interval: Steps between logging
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for logs
        early_stopping: Whether to use early stopping
        patience: Early stopping patience
        min_delta: Minimum improvement for early stopping
    """
    algorithm: str = "ppo"
    total_steps: int = 100000
    rollout_steps: int = 2048
    eval_interval: int = 1000
    save_interval: int = 5000
    log_interval: int = 100
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.001


@dataclass
class RolloutData:
    """Data collected during rollouts.

    Attributes:
        observations: List of observations
        actions: List of actions
        rewards: List of rewards
        log_probs: List of log probabilities
        values: List of value estimates
        dones: List of done flags
        masks: List of continuation masks
        infos: List of info dicts
    """
    observations: List[Any] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    masks: List[float] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self) -> None:
        """Clear all data."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.masks = []
        self.infos = []

    def to_tensor_dict(
        self,
        obs_to_tensor: Optional[Callable] = None,
        action_to_tensor: Optional[Callable] = None,
        device: Union[str, torch.device] = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """Convert to dictionary of tensors.

        Args:
            obs_to_tensor: Function to convert observations to tensors
            action_to_tensor: Function to convert actions to tensors
            device: Target device

        Returns:
            Dictionary of tensors
        """
        # Convert observations
        if obs_to_tensor is not None:
            observations = obs_to_tensor(self.observations)
        else:
            observations = torch.tensor(np.array(self.observations), device=device)

        # Convert actions
        if action_to_tensor is not None:
            actions = action_to_tensor(self.actions)
        else:
            actions = torch.tensor(np.array(self.actions), device=device)

        return {
            "observations": observations,
            "actions": actions,
            "rewards": torch.tensor(self.rewards, dtype=torch.float32, device=device),
            "masks": torch.tensor(self.masks, dtype=torch.float32, device=device),
            "log_probs": torch.tensor(self.log_probs, dtype=torch.float32, device=device),
            "values": torch.tensor(self.values, dtype=torch.float32, device=device),
        }


@dataclass
class TrainingMetrics:
    """Metrics tracked during training.

    Attributes:
        step: Current training step
        episode: Current episode number
        reward_mean: Mean episode reward
        reward_std: Std of episode reward
        episode_length: Mean episode length
        loss: Current loss value
        learning_rate: Current learning rate
    """
    step: int = 0
    episode: int = 0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    episode_length: float = 0.0
    loss: float = 0.0
    learning_rate: float = 0.0
    kl_divergence: float = 0.0
    entropy: float = 0.0


class RLTrainer:
    """Main RL Trainer for LLM Agents.

    This class orchestrates the training process, including:
    - Rollout collection
    - Policy updates
    - Evaluation
    - Checkpointing
    - Logging

    Usage:
        config = TrainingConfig(algorithm='ppo', total_steps=100000)
        trainer = RLTrainer(policy, env, config)
        trainer.train()
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        env: Any,
        config: Optional[TrainingConfig] = None,
        ppo_config: Optional[PPOConfig] = None,
        grpo_config: Optional[GRPOConfig] = None,
        device: Union[str, torch.device] = "cpu",
        obs_to_tensor: Optional[Callable] = None,
        action_to_tensor: Optional[Callable] = None,
    ):
        """Initialize RL trainer.

        Args:
            policy: Policy model
            env: Environment or environment factory
            config: Training configuration
            ppo_config: PPO-specific configuration
            grpo_config: GRPO-specific configuration
            device: Training device
            obs_to_tensor: Function to convert observations to tensors
            action_to_tensor: Function to convert actions to tensors
        """
        self.policy = policy
        self.env = env
        self.config = config or TrainingConfig()
        self.device = torch.device(device)

        # Conversion functions
        self.obs_to_tensor = obs_to_tensor
        self.action_to_tensor = action_to_tensor

        # Initialize algorithm-specific trainer
        if self.config.algorithm == "ppo":
            self.algo_config = ppo_config or PPOConfig()
            self.algo_trainer = PPOTrainer(
                policy=self.policy,
                config=self.algo_config,
                device=self.device,
            )
        elif self.config.algorithm == "grpo":
            self.algo_config = grpo_config or GRPOConfig()
            self.algo_trainer = GRPOTrainer(
                policy=self.policy,
                config=self.algo_config,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

        # Rollout buffer
        self.rollout_buffer = RolloutData()

        # TensorBoard writer
        self.writer: Optional[SummaryWriter] = None

        # Training state
        self._current_step = 0
        self._current_episode = 0
        self._best_eval_reward = float("-inf")
        self._patience_counter = 0
        self._training_history: List[TrainingMetrics] = []

        # Create directories
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

    def _init_writer(self) -> None:
        """Initialize TensorBoard writer."""
        log_path = os.path.join(
            self.config.log_dir,
            datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.writer = SummaryWriter(log_path)

    def collect_rollout(
        self,
        num_steps: Optional[int] = None,
        deterministic: bool = False
    ) -> RolloutData:
        """Collect rollout data from the environment.

        Args:
            num_steps: Number of steps to collect (defaults to config.rollout_steps)
            deterministic: Whether to use deterministic actions

        Returns:
            RolloutData containing collected experience
        """
        if num_steps is None:
            num_steps = self.config.rollout_steps

        rollout = RolloutData()
        obs = self.env.reset()

        steps_collected = 0

        while steps_collected < num_steps:
            # Convert observation to tensor
            if self.obs_to_tensor is not None:
                obs_tensor = self.obs_to_tensor(obs)
            else:
                obs_tensor = torch.tensor([obs], device=self.device)

            # Get action from policy
            with torch.no_grad():
                if deterministic:
                    action = self.policy.get_action(obs_tensor, deterministic=True)
                else:
                    action, log_prob = self.policy.get_action_with_log_prob(obs_tensor)

            # Convert action to numpy if needed
            action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action

            # Take step in environment
            next_obs, reward, done, truncated, info = self.env.step(action_np)

            # Get value estimate
            with torch.no_grad():
                value = self.policy.get_value(obs_tensor)

            # Store transition
            rollout.observations.append(obs)
            rollout.actions.append(action_np)
            rollout.rewards.append(reward)
            rollout.log_probs.append(log_prob.cpu().numpy() if isinstance(log_prob, torch.Tensor) else log_prob)
            rollout.values.append(value.cpu().numpy().flatten()[0] if isinstance(value, torch.Tensor) else value)
            rollout.dones.append(done or truncated)
            rollout.masks.append(0.0 if done or truncated else 1.0)
            rollout.infos.append(info)

            steps_collected += 1
            self._current_step += 1

            # Handle episode end
            if done or truncated:
                self._current_episode += 1
                obs = self.env.reset()
            else:
                obs = next_obs

        return rollout

    def update_policy(self, rollout: RolloutData) -> Union[PPOStats, GRPOStats]:
        """Update policy using collected rollout.

        Args:
            rollout: Rollout data to train on

        Returns:
            Training statistics
        """
        # Convert to tensor dict
        tensor_dict = rollout.to_tensor_dict(
            obs_to_tensor=self.obs_to_tensor,
            action_to_tensor=self.action_to_tensor,
            device=self.device,
        )

        # Update policy
        stats = self.algo_trainer.update(tensor_dict)

        return stats

    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """Evaluate the current policy.

        Args:
            num_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic actions

        Returns:
            Dictionary of evaluation metrics
        """
        rewards = []
        episode_lengths = []
        success_rates = []

        for _ in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0.0
            episode_length = 0

            while True:
                if self.obs_to_tensor is not None:
                    obs_tensor = self.obs_to_tensor(obs)
                else:
                    obs_tensor = torch.tensor([obs], device=self.device)

                with torch.no_grad():
                    if deterministic:
                        action = self.policy.get_action(obs_tensor, deterministic=True)
                    else:
                        action, _ = self.policy.get_action_with_log_prob(obs_tensor)

                action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
                obs, reward, done, truncated, info = self.env.step(action_np)

                episode_reward += reward
                episode_length += 1

                if done or truncated:
                    break

            rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if "success" in info or "correct" in info or "goal_reached" in info:
                success_rates.append(
                    1.0 if any(info.get(k, False) for k in ["success", "correct", "goal_reached"]) else 0.0
                )

        metrics = {
            "eval_reward_mean": np.mean(rewards),
            "eval_reward_std": np.std(rewards),
            "eval_episode_length": np.mean(episode_lengths),
        }

        if success_rates:
            metrics["eval_success_rate"] = np.mean(success_rates)

        return metrics

    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        verbose: bool = True
    ) -> List[TrainingMetrics]:
        """Main training loop.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
            verbose: Whether to print progress

        Returns:
            Training history
        """
        # Initialize writer
        self._init_writer()

        # Load checkpoint if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)

        # Training loop
        pbar = tqdm(
            range(self._current_step, self.config.total_steps),
            desc="Training",
            disable=not verbose,
        )

        for step in pbar:
            # Collect rollout
            rollout = self.collect_rollout()

            # Update policy
            stats = self.update_policy(rollout)

            # Update metrics
            metrics = TrainingMetrics(
                step=step,
                episode=self._current_episode,
                reward_mean=np.mean(rollout.rewards),
                loss=stats.policy_loss if hasattr(stats, 'policy_loss') else 0.0,
                learning_rate=stats.learning_rate if hasattr(stats, 'learning_rate') else 0.0,
                kl_divergence=stats.kl_divergence if hasattr(stats, 'kl_divergence') else 0.0,
                entropy=stats.entropy if hasattr(stats, 'entropy') else 0.0,
            )

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar("loss/policy", stats.policy_loss, step)
                self.writer.add_scalar("loss/value", stats.value_loss, step)
                self.writer.add_scalar("metrics/entropy", stats.entropy, step)
                self.writer.add_scalar("metrics/kl", stats.kl_divergence, step)
                self.writer.add_scalar("metrics/reward_mean", metrics.reward_mean, step)
                self.writer.add_scalar("metrics/learning_rate", metrics.learning_rate, step)

            # Evaluation
            if step % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()

                if self.writer:
                    for key, value in eval_metrics.items():
                        self.writer.add_scalar(f"eval/{key}", value, step)

                # Early stopping check
                if self.config.early_stopping:
                    eval_reward = eval_metrics.get("eval_reward_mean", 0.0)
                    if eval_reward > self._best_eval_reward + self.config.min_delta:
                        self._best_eval_reward = eval_reward
                        self._patience_counter = 0
                        # Save best model
                        self.save_checkpoint(
                            os.path.join(self.config.checkpoint_dir, "best_model.pt")
                        )
                    else:
                        self._patience_counter += 1
                        if self._patience_counter >= self.config.patience:
                            if verbose:
                                print(f"Early stopping at step {step}")
                            break

            # Save checkpoint
            if step % self.config.save_interval == 0:
                self.save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, f"checkpoint_{step}.pt")
                )

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{stats.policy_loss:.4f}",
                "reward": f"{metrics.reward_mean:.4f}",
            })

            self._training_history.append(metrics)

        self.writer.close()
        return self._training_history

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "algo_trainer_state": self.algo_trainer.__class__.__name__,
            "current_step": self._current_step,
            "current_episode": self._current_episode,
            "best_eval_reward": self._best_eval_reward,
            "config": self.config,
            "training_history": self._training_history,
        }

        # Save algorithm-specific state
        if isinstance(self.algo_trainer, PPOTrainer):
            checkpoint["algo_state_dict"] = {
                "optimizer": self.algo_trainer.optimizer.state_dict(),
            }
        elif isinstance(self.algo_trainer, GRPOTrainer):
            checkpoint["algo_state_dict"] = {
                "optimizer": self.algo_trainer.optimizer.state_dict(),
            }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self._current_step = checkpoint["current_step"]
        self._current_episode = checkpoint["current_episode"]
        self._best_eval_reward = checkpoint["best_eval_reward"]
        self._training_history = checkpoint.get("training_history", [])

        if "algo_state_dict" in checkpoint:
            if isinstance(self.algo_trainer, (PPOTrainer, GRPOTrainer)):
                self.algo_trainer.optimizer.load_state_dict(
                    checkpoint["algo_state_dict"]["optimizer"]
                )

    def get_training_history(self) -> List[TrainingMetrics]:
        """Get training history.

        Returns:
            List of TrainingMetrics
        """
        return self._training_history
