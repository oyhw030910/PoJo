"""Loss Functions for RL.

Implements loss functions for PPO, GRPO and other RL algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LossOutput:
    """Output of loss computation.

    Attributes:
        loss: Total loss value
        policy_loss: Policy loss component
        value_loss: Value loss component
        entropy: Entropy bonus
        metrics: Additional metrics
    """
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    metrics: Dict[str, float]


class PolicyLoss(nn.Module):
    """PPO Policy Loss.

    Implements the PPO-Clip policy loss:
    L^CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]

    where r_t = pi_new(a|s) / pi_old(a|s)
    """

    def __init__(self, clip_epsilon: float = 0.2):
        super().__init__()
        self.clip_epsilon = clip_epsilon

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute policy loss.

        Args:
            log_probs: Current policy log probabilities
            old_log_probs: Old policy log probabilities (behavior policy)
            advantages: Advantage estimates
            mask: Optional mask for padding

        Returns:
            Tuple of (policy_loss, clip_fraction)
        """
        # Compute importance sampling ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Compute clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

        # Take minimum of clipped and unclipped
        policy_loss = -torch.min(surr1, surr2)

        # Apply mask if provided
        if mask is not None:
            policy_loss = policy_loss * mask
            policy_loss = policy_loss.sum() / mask.sum()
        else:
            policy_loss = policy_loss.mean()

        # Compute clip fraction (how often clipping occurs)
        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_epsilon).float())

        return policy_loss, clip_fraction


class ValueLoss(nn.Module):
    """Value Function Loss.

    Implements MSE loss for value function with optional clipping.
    """

    def __init__(self, clip_value_loss: bool = True):
        super().__init__()
        self.clip_value_loss = clip_value_loss

    def forward(
        self,
        values: torch.Tensor,
        targets: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute value loss.

        Args:
            values: Current value predictions
            targets: Target values (returns)
            old_values: Old value predictions (for clipping)
            mask: Optional mask for padding

        Returns:
            Value loss
        """
        if self.clip_value_loss and old_values is not None:
            # Clip value predictions to prevent large updates
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -self.clip_epsilon if hasattr(self, 'clip_epsilon') else 0.2,
                self.clip_epsilon if hasattr(self, 'clip_epsilon') else 0.2
            )
            loss1 = (values - targets) ** 2
            loss2 = (values_clipped - targets) ** 2
            value_loss = torch.max(loss1, loss2)
        else:
            value_loss = (values - targets) ** 2

        # Apply mask if provided
        if mask is not None:
            value_loss = value_loss * mask
            value_loss = value_loss.sum() / mask.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss


class EntropyBonus(nn.Module):
    """Entropy Bonus for Exploration.

    Adds entropy bonus to encourage exploration.
    """

    def __init__(self, coef: float = 0.01):
        super().__init__()
        self.coef = coef

    def forward(
        self,
        log_probs: torch.Tensor,
        probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute entropy bonus.

        Args:
            log_probs: Log probabilities
            probs: Optional probabilities (if None, computed from log_probs)

        Returns:
            Entropy bonus (to be added to loss, so negative entropy)
        """
        if probs is not None:
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        else:
            # Approximate entropy from log_probs
            # This assumes uniform distribution over non-zero probs
            entropy = -log_probs.mean()

        # Return negative entropy (to be added as bonus)
        return -self.coef * entropy.mean()


class KLPenalty(nn.Module):
    """KL Divergence Penalty.

    Computes KL divergence between two policies.
    """

    def __init__(self, coef: float = 0.2):
        super().__init__()
        self.coef = coef

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """Compute KL penalty.

        Args:
            log_probs: Current policy log probabilities
            old_log_probs: Reference policy log probabilities
            mask: Optional mask

        Returns:
            Tuple of (kl_penalty, kl_mean)
        """
        # KL = log(pi_new/pi_old) = log_probs - old_log_probs
        # But we need E[pi_new * log(pi_new/pi_old)]
        # Approximation: use log_probs as weights
        kl = old_log_probs - log_probs  # Note: opposite sign for penalty

        if mask is not None:
            kl_penalty = (kl * mask).sum() / mask.sum()
        else:
            kl_penalty = kl.mean()

        kl_mean = kl_penalty.item() if isinstance(kl_penalty, torch.Tensor) else kl_penalty

        return self.coef * kl_penalty, kl_mean


class GAE(nn.Module):
    """Generalized Advantage Estimation.

    Computes advantage estimates using GAE.
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        super().__init__()
        self.gamma = gamma
        self.lam = lam

    def forward(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        masks: torch.Tensor,
        truncate_last: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns.

        Args:
            rewards: Rewards at each timestep
            values: Value estimates at each timestep
            next_values: Value estimates for next states
            masks: Done masks (0 if done, 1 otherwise)
            truncate_last: Whether to truncate last timestep

        Returns:
            Tuple of (advantages, returns)
        """
        batch_size, seq_len = rewards.shape[:2]

        # Compute TD residuals
        deltas = rewards + self.gamma * next_values * masks - values

        # Initialize
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Backward pass
        last_advantage = torch.zeros(batch_size, device=rewards.device)
        last_return = torch.zeros(batch_size, device=rewards.device)

        for t in reversed(range(seq_len)):
            last_advantage = deltas[:, t] + self.gamma * self.lam * masks[:, t] * last_advantage
            last_return = rewards[:, t] + self.gamma * masks[:, t] * last_return

            advantages[:, t] = last_advantage
            returns[:, t] = last_return

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns


class GRPOLoss(nn.Module):
    """GRPO (Group Relative Policy Optimization) Loss.

    Implements group-based advantage computation without value function.
    """

    def __init__(self, clip_epsilon: float = 0.2, group_size: int = 8):
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.group_size = group_size

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        group_rewards: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GRPO loss.

        Args:
            log_probs: Current policy log probabilities
            old_log_probs: Old policy log probabilities
            rewards: Rewards for each sample
            group_rewards: Optional pre-computed group rewards
            mask: Optional mask

        Returns:
            Tuple of (policy_loss, advantage_stats)
        """
        batch_size = log_probs.shape[0]

        # Compute group-based advantages
        if group_rewards is None:
            # Reshape into groups
            num_groups = batch_size // self.group_size
            if num_groups > 0:
                reshaped_rewards = rewards[:num_groups * self.group_size].view(
                    num_groups, self.group_size
                )
                # Compute group mean and std
                group_mean = reshaped_rewards.mean(dim=1, keepdim=True)
                group_std = reshaped_rewards.std(dim=1, keepdim=True) + 1e-8
                # Normalize within group
                normalized_rewards = (reshaped_rewards - group_mean) / group_std
                advantages = normalized_rewards.view(-1)
                # Pad if necessary
                if num_groups * self.group_size < batch_size:
                    remaining = batch_size - num_groups * self.group_size
                    advantages = torch.cat([
                        advantages,
                        torch.zeros(remaining, device=rewards.device)
                    ])
            else:
                # Fallback to simple normalization
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = group_rewards

        # Expand advantages to match sequence length
        if len(advantages.shape) < len(log_probs.shape):
            advantages = advantages.unsqueeze(1).expand_as(log_probs)

        # Compute policy loss (similar to PPO)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2)

        if mask is not None:
            policy_loss = policy_loss * mask
            policy_loss = policy_loss.sum() / mask.sum()
        else:
            policy_loss = policy_loss.mean()

        # Clip fraction
        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_epsilon).float())

        return policy_loss, clip_fraction


class LossCollector:
    """Collects and aggregates loss metrics."""

    def __init__(self):
        self.metrics: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, name: str, value: float, count: int = 1) -> None:
        """Update a metric.

        Args:
            name: Metric name
            value: Metric value
            count: Number of samples
        """
        if name not in self.metrics:
            self.metrics[name] = 0.0
            self.counts[name] = 0
        self.metrics[name] += value * count
        self.counts[name] += count

    def get_average(self, name: str) -> float:
        """Get average value for a metric.

        Args:
            name: Metric name

        Returns:
            Average value
        """
        if name not in self.counts or self.counts[name] == 0:
            return 0.0
        return self.metrics[name] / self.counts[name]

    def get_all_averages(self) -> Dict[str, float]:
        """Get all averaged metrics.

        Returns:
            Dictionary of metric averages
        """
        return {name: self.get_average(name) for name in self.metrics}

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}
