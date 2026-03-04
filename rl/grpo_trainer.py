"""GRPO Trainer Module.

Implements Group Relative Policy Optimization (GRPO) algorithm for RL.
GRPO eliminates the need for a value function by using group-relative advantages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm

from .loss import GRPOLoss, EntropyBonus


@dataclass
class GRPOConfig:
    """Configuration for GRPO training.

    Attributes:
        lr: Learning rate
        betas: Adam betas
        eps: Adam epsilon
        weight_decay: Weight decay
        epochs: GRPO epochs per update
        batch_size: Training batch size
        mini_batch_size: Mini-batch size
        gamma: Discount factor
        clip_epsilon: GRPO clip epsilon
        group_size: Number of samples per group
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm
        use_baseline: Whether to use group baseline
    """
    lr: float = 3e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    epochs: int = 10
    batch_size: int = 64
    mini_batch_size: int = 32
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    group_size: int = 8
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    use_baseline: bool = True


@dataclass
class GRPOStats:
    """Statistics from GRPO update.

    Attributes:
        policy_loss: Average policy loss
        entropy: Average entropy
        group_advantage_mean: Mean group advantage
        group_advantage_std: Std of group advantage
        clip_fraction: Fraction of clipped updates
        learning_rate: Current learning rate
    """
    policy_loss: float = 0.0
    entropy: float = 0.0
    group_advantage_mean: float = 0.0
    group_advantage_std: float = 0.0
    clip_fraction: float = 0.0
    learning_rate: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0


@dataclass
class GroupSample:
    """A group of samples for GRPO.

    Attributes:
        observations: Group observations
        actions: Group actions
        rewards: Group rewards
        log_probs: Group log probabilities
    """
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    log_probs: torch.Tensor


class GRPOTrainer:
    """GRPO Trainer for RL optimization.

    Group Relative Policy Optimization (GRPO) is a variant of PPO that
    eliminates the need for a value function by computing advantages
    relative to the group mean.

    Key idea:
    - Sample a group of outputs from the same input
    - Compute rewards for each output
    - Normalize rewards within group to get advantages

    Usage:
        config = GRPOConfig(group_size=8, epochs=10)
        trainer = GRPOTrainer(policy_model, config)

        rollouts = collect_rollouts(env, policy)
        stats = trainer.update(rollouts)
    """

    def __init__(
        self,
        policy: nn.Module,
        config: Optional[GRPOConfig] = None,
        device: Union[str, torch.device] = "cpu"
    ):
        """Initialize GRPO trainer.

        Args:
            policy: Policy model
            config: GRPO configuration
            device: Training device
        """
        self.policy = policy
        self.config = config or GRPOConfig()
        self.device = torch.device(device)

        # Move policy to device
        self.policy.to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )

        # Loss functions
        self.grpo_loss_fn = GRPOLoss(
            clip_epsilon=self.config.clip_epsilon,
            group_size=self.config.group_size,
        )
        self.entropy_bonus_fn = EntropyBonus(coef=self.config.entropy_coef)

        # Statistics
        self._stats_history: List[GRPOStats] = []

    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
        group_size: Optional[int] = None
    ) -> torch.Tensor:
        """Compute group-relative advantages.

        Args:
            rewards: Rewards tensor [batch_size, seq_len]
            group_size: Size of each group

        Returns:
            Advantages tensor with same shape as rewards
        """
        if group_size is None:
            group_size = self.config.group_size

        batch_size = rewards.shape[0]
        num_groups = batch_size // group_size

        if num_groups == 0:
            # Fallback: simple normalization
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            return advantages

        # Reshape to [num_groups, group_size, ...]
        trimmed_size = num_groups * group_size
        trimmed_rewards = rewards[:trimmed_size]
        reshaped = trimmed_rewards.view(num_groups, group_size, *rewards.shape[1:])

        # Compute group statistics
        if self.config.use_baseline:
            group_mean = reshaped.mean(dim=1, keepdim=True)
            group_std = reshaped.std(dim=1, keepdim=True) + 1e-8

            # Normalize within group
            advantages_normalized = (reshaped - group_mean) / group_std
        else:
            # Just use raw rewards as advantages
            advantages_normalized = reshaped

        # Reshape back
        advantages = advantages_normalized.view(trimmed_size, *rewards.shape[1:])

        # Pad if necessary
        if trimmed_size < batch_size:
            remaining = batch_size - trimmed_size
            padding = torch.zeros(remaining, *rewards.shape[1:], device=rewards.device)
            advantages = torch.cat([advantages, padding], dim=0)

        return advantages

    def update(
        self,
        rollouts: Dict[str, torch.Tensor],
        verbose: bool = False
    ) -> GRPOStats:
        """Update policy using GRPO.

        Args:
            rollouts: Dictionary containing:
                - observations: [B, T, ...]
                - actions: [B, T]
                - rewards: [B, T]
                - masks: [B, T]
                - log_probs: [B, T]
            verbose: Whether to show progress bar

        Returns:
            GRPOStats with training statistics
        """
        # Extract rollout data
        observations = rollouts["observations"].to(self.device)
        actions = rollouts["actions"].to(self.device)
        rewards = rollouts["rewards"].to(self.device)
        masks = rollouts.get("masks", torch.ones_like(rewards)).to(self.device)
        old_log_probs = rollouts["log_probs"].to(self.device)

        # Compute group advantages
        advantages = self.compute_group_advantages(rewards)

        # Create dataset
        dataset = TensorDataset(
            observations,
            actions,
            old_log_probs,
            advantages,
            rewards,
            masks,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.mini_batch_size,
            shuffle=True,
        )

        # Training statistics
        total_policy_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        total_advantage_mean = 0.0
        total_advantage_std = 0.0
        num_updates = 0

        # GRPO epochs
        for epoch in range(self.config.epochs):
            epoch_policy_loss = 0.0
            epoch_entropy = 0.0
            epoch_clip_frac = 0.0
            epoch_updates = 0

            iterator = dataloader
            if verbose:
                iterator = tqdm(dataloader, desc=f"GRPO Epoch {epoch + 1}/{self.config.epochs}")

            for batch in iterator:
                (
                    batch_obs,
                    batch_actions,
                    batch_old_log_probs,
                    batch_advantages,
                    batch_rewards,
                    batch_masks,
                ) = batch

                # Move to device
                batch_obs = batch_obs.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_old_log_probs = batch_old_log_probs.to(self.device)
                batch_advantages = batch_advantages.to(self.device)
                batch_masks = batch_masks.to(self.device)

                # Forward pass
                outputs = self.policy.forward_for_training(
                    batch_obs,
                    batch_actions,
                )
                log_probs = outputs["log_probs"]
                entropy = outputs.get("entropy", torch.zeros(1)).to(self.device)

                # GRPO loss
                policy_loss, clip_frac = self.grpo_loss_fn(
                    log_probs=log_probs,
                    old_log_probs=batch_old_log_probs,
                    rewards=batch_rewards,
                    group_rewards=batch_advantages,
                    mask=batch_masks,
                )

                # Entropy bonus
                entropy_bonus = self.entropy_bonus_fn(log_probs)

                # Total loss
                loss = policy_loss + entropy_bonus

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm,
                )

                # Update
                self.optimizer.step()

                # Accumulate statistics
                epoch_policy_loss += policy_loss.item()
                epoch_entropy += entropy.mean().item()
                epoch_clip_frac += clip_frac.item()
                epoch_updates += 1

            # Epoch statistics
            total_policy_loss += epoch_policy_loss / epoch_updates
            total_entropy += epoch_entropy / epoch_updates
            total_clip_frac += epoch_clip_frac / epoch_updates
            num_updates += 1

            # Advantage statistics
            total_advantage_mean += advantages.mean().item()
            total_advantage_std += advantages.std().item()

        # Compute final statistics
        stats = GRPOStats(
            policy_loss=total_policy_loss / num_updates,
            entropy=total_entropy / num_updates,
            clip_fraction=total_clip_frac / num_updates,
            group_advantage_mean=total_advantage_mean / num_updates,
            group_advantage_std=total_advantage_std / num_updates,
            learning_rate=self.optimizer.param_groups[0]["lr"],
            reward_mean=rewards.mean().item(),
            reward_std=rewards.std().item(),
        )

        self._stats_history.append(stats)

        return stats

    def update_learning_rate(self, lr: float) -> None:
        """Update learning rate.

        Args:
            lr: New learning rate
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_stats_history(self) -> List[GRPOStats]:
        """Get training statistics history.

        Returns:
            List of GRPOStats
        """
        return self._stats_history

    def clear_stats(self) -> None:
        """Clear statistics history."""
        self._stats_history = []

    def save_checkpoint(self, path: str) -> None:
        """Save trainer checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "stats_history": self._stats_history,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load trainer checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._stats_history = checkpoint.get("stats_history", [])
