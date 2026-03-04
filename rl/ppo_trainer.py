"""PPO Trainer Module.

Implements Proximal Policy Optimization (PPO) algorithm for RL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm

from .loss import PolicyLoss, ValueLoss, EntropyBonus, KLPenalty, GAE, LossOutput


@dataclass
class PPOConfig:
    """Configuration for PPO training.

    Attributes:
        lr: Learning rate
        betas: Adam betas
        eps: Adam epsilon
        weight_decay: Weight decay
        epochs: PPO epochs per update
        batch_size: Training batch size
        mini_batch_size: Mini-batch size for gradient accumulation
        gamma: Discount factor
        lam: GAE lambda
        clip_epsilon: PPO clip epsilon
        clip_value_loss: Whether to clip value loss
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm
        target_kl: Target KL divergence for early stopping
        use_kl_penalty: Whether to use KL penalty
        kl_coef: KL penalty coefficient
    """
    lr: float = 3e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    epochs: int = 10
    batch_size: int = 64
    mini_batch_size: int = 32
    gamma: float = 0.99
    lam: float = 0.95
    clip_epsilon: float = 0.2
    clip_value_loss: bool = True
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    use_kl_penalty: bool = False
    kl_coef: float = 0.2


@dataclass
class PPOBatch:
    """Batch data for PPO training.

    Attributes:
        observations: Batch observations
        actions: Batch actions
        old_log_probs: Old log probabilities
        advantages: Advantage estimates
        returns: Return estimates
        values: Old value estimates
        masks: Attention masks (for transformers)
    """
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    masks: Optional[torch.Tensor] = None


@dataclass
class PPOStats:
    """Statistics from PPO update.

    Attributes:
        policy_loss: Average policy loss
        value_loss: Average value loss
        entropy: Average entropy
        kl_divergence: Average KL divergence
        clip_fraction: Fraction of clipped updates
        explained_variance: Explained variance of value function
        approx_kl: Approximate KL divergence
    """
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    kl_divergence: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0
    approx_kl: float = 0.0
    learning_rate: float = 0.0


class PPOTrainer:
    """PPO Trainer for RL optimization.

    This class implements the Proximal Policy Optimization algorithm
    with support for transformer-based policies.

    Usage:
        config = PPOConfig(lr=3e-5, epochs=10, batch_size=64)
        trainer = PPOTrainer(policy_model, config)

        # Collect rollouts
        rollouts = collect_rollouts(env, policy)

        # Update policy
        stats = trainer.update(rollouts)
    """

    def __init__(
        self,
        policy: nn.Module,
        config: Optional[PPOConfig] = None,
        device: Union[str, torch.device] = "cpu"
    ):
        """Initialize PPO trainer.

        Args:
            policy: Policy model (with get_value and get_log_probs methods)
            config: PPO configuration
            device: Training device
        """
        self.policy = policy
        self.config = config or PPOConfig()
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

        # Learning rate scheduler
        self.scheduler = None

        # Loss functions
        self.policy_loss_fn = PolicyLoss(clip_epsilon=self.config.clip_epsilon)
        self.value_loss_fn = ValueLoss(clip_value_loss=self.config.clip_value_loss)
        self.entropy_bonus_fn = EntropyBonus(coef=self.config.entropy_coef)
        self.kl_penalty_fn = KLPenalty(coef=self.config.kl_coef)
        self.gae_fn = GAE(gamma=self.config.gamma, lam=self.config.lam)

        # Statistics
        self._stats_history: List[PPOStats] = []

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        next_values: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns.

        Args:
            rewards: Rewards tensor
            values: Value estimates
            masks: Done masks
            next_values: Next state values (for bootstrapping)

        Returns:
            Tuple of (advantages, returns)
        """
        if next_values is None:
            next_values = torch.zeros_like(values[:, -1:])

        # Ensure shapes match
        if len(next_values.shape) < 2:
            next_values = next_values.unsqueeze(1)

        # Concatenate for bootstrap
        values_with_bootstrap = torch.cat([values, next_values], dim=1)
        current_values = values_with_bootstrap[:, :-1]
        next_vals = values_with_bootstrap[:, 1:]

        # Compute advantages
        advantages, returns = self.gae_fn(
            rewards=rewards,
            values=current_values,
            next_values=next_vals,
            masks=masks,
        )

        return advantages, returns

    def update(
        self,
        rollouts: Dict[str, torch.Tensor],
        verbose: bool = False
    ) -> PPOStats:
        """Update policy using PPO.

        Args:
            rollouts: Dictionary containing:
                - observations: [B, T, ...]
                - actions: [B, T]
                - rewards: [B, T]
                - masks: [B, T]
                - log_probs: [B, T]
                - values: [B, T]
            verbose: Whether to show progress bar

        Returns:
            PPOStats with training statistics
        """
        # Extract rollout data
        observations = rollouts["observations"].to(self.device)
        actions = rollouts["actions"].to(self.device)
        rewards = rollouts["rewards"].to(self.device)
        masks = rollouts.get("masks", torch.ones_like(rewards)).to(self.device)
        old_log_probs = rollouts["log_probs"].to(self.device)
        old_values = rollouts.get("values", torch.zeros_like(rewards)).to(self.device)

        # Compute advantages and returns
        advantages, returns = self.compute_advantages(
            rewards=rewards,
            values=old_values,
            masks=masks,
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create dataset and dataloader
        dataset = TensorDataset(
            observations,
            actions,
            old_log_probs,
            advantages,
            returns,
            old_values,
            masks,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.mini_batch_size,
            shuffle=True,
        )

        # Training statistics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        num_updates = 0

        # PPO epochs
        for epoch in range(self.config.epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            epoch_kl = 0.0
            epoch_clip_frac = 0.0
            epoch_updates = 0

            # Progress bar for epoch
            iterator = dataloader
            if verbose:
                iterator = tqdm(dataloader, desc=f"PPO Epoch {epoch + 1}/{self.config.epochs}")

            for batch in iterator:
                (
                    batch_obs,
                    batch_actions,
                    batch_old_log_probs,
                    batch_advantages,
                    batch_returns,
                    batch_old_values,
                    batch_masks,
                ) = batch

                # Move to device
                batch_obs = batch_obs.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_old_log_probs = batch_old_log_probs.to(self.device)
                batch_advantages = batch_advantages.to(self.device)
                batch_returns = batch_returns.to(self.device)
                batch_old_values = batch_old_values.to(self.device)
                batch_masks = batch_masks.to(self.device)

                # Forward pass through policy
                outputs = self.policy.forward_for_training(
                    batch_obs,
                    batch_actions,
                )
                log_probs = outputs["log_probs"]  # [B, T]
                values = outputs["values"]  # [B, T]
                entropy = outputs.get("entropy", torch.zeros(1)).to(self.device)

                # Policy loss
                policy_loss, clip_frac = self.policy_loss_fn(
                    log_probs=log_probs,
                    old_log_probs=batch_old_log_probs,
                    advantages=batch_advantages,
                    mask=batch_masks,
                )

                # Value loss
                value_loss = self.value_loss_fn(
                    values=values,
                    targets=batch_returns,
                    old_values=batch_old_values,
                    mask=batch_masks,
                )

                # Entropy bonus
                entropy_bonus = self.entropy_bonus_fn(log_probs)

                # KL penalty (optional)
                kl_penalty = torch.tensor(0.0, device=self.device)
                if self.config.use_kl_penalty:
                    kl_penalty, _ = self.kl_penalty_fn(
                        log_probs=log_probs,
                        old_log_probs=batch_old_log_probs,
                        mask=batch_masks,
                    )

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + entropy_bonus
                    + kl_penalty
                )

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
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy.mean().item()
                epoch_clip_frac += clip_frac.item()
                epoch_updates += 1

                # Approximate KL
                approx_kl = (batch_old_log_probs - log_probs).mean().item()
                epoch_kl += approx_kl

            # Epoch statistics
            total_policy_loss += epoch_policy_loss / epoch_updates
            total_value_loss += epoch_value_loss / epoch_updates
            total_entropy += epoch_entropy / epoch_updates
            total_kl += epoch_kl / epoch_updates
            total_clip_frac += epoch_clip_frac / epoch_updates
            num_updates += 1

            # Early stopping on KL
            if self.config.target_kl > 0 and (epoch_kl / epoch_updates) > self.config.target_kl:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1} due to KL divergence")
                break

        # Compute final statistics
        stats = PPOStats(
            policy_loss=total_policy_loss / num_updates,
            value_loss=total_value_loss / num_updates,
            entropy=total_entropy / num_updates,
            kl_divergence=total_kl / num_updates,
            clip_fraction=total_clip_frac / num_updates,
            approx_kl=total_kl / num_updates,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )

        # Compute explained variance
        with torch.no_grad():
            pred_values = self.policy.get_value(observations)
            if pred_values.shape != returns.shape:
                pred_values = pred_values.squeeze(-1)
            var_pred = torch.var(pred_values)
            var_true = torch.var(returns)
            cov = torch.mean((pred_values - pred_values.mean()) * (returns - returns.mean()))
            if var_true > 0:
                stats.explained_variance = (cov / (var_true + 1e-8)).item()
            else:
                stats.explained_variance = 0.0

        self._stats_history.append(stats)

        return stats

    def update_learning_rate(self, lr: float) -> None:
        """Update learning rate.

        Args:
            lr: New learning rate
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_stats_history(self) -> List[PPOStats]:
        """Get training statistics history.

        Returns:
            List of PPOStats
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
