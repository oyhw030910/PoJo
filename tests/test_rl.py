"""Tests for RL Module."""

import pytest
import torch
import numpy as np

from rl.loss import PolicyLoss, ValueLoss, EntropyBonus, GAE, GRPOLoss
from rl.replay_buffer import ReplayBuffer, SequenceBuffer, PriorityReplayBuffer
from rl.ppo_trainer import PPOTrainer, PPOConfig
from rl.grpo_trainer import GRPOTrainer, GRPOConfig


class TestPolicyLoss:
    """Tests for policy loss."""

    def test_policy_loss_basic(self):
        """Test basic policy loss computation."""
        loss_fn = PolicyLoss(clip_epsilon=0.2)

        log_probs = torch.tensor([0.1, 0.2, 0.3])
        old_log_probs = torch.tensor([0.0, 0.1, 0.2])
        advantages = torch.tensor([1.0, 1.0, 1.0])

        loss, clip_frac = loss_fn(log_probs, old_log_probs, advantages)

        assert loss.item() < 0  # Policy loss should be negative (to be minimized)
        assert 0 <= clip_frac.item() <= 1

    def test_policy_loss_clipping(self):
        """Test policy loss clipping."""
        loss_fn = PolicyLoss(clip_epsilon=0.2)

        # Large ratio should be clipped
        log_probs = torch.tensor([1.0])  # exp(1) = 2.72, much larger than 1.2
        old_log_probs = torch.tensor([0.0])
        advantages = torch.tensor([1.0])

        loss, clip_frac = loss_fn(log_probs, old_log_probs, advantages)

        assert clip_frac.item() == 1.0  # All clipped


class TestValueLoss:
    """Tests for value loss."""

    def test_value_loss_basic(self):
        """Test basic value loss computation."""
        loss_fn = ValueLoss(clip_value_loss=True)

        values = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.5, 2.5, 3.5])

        loss = loss_fn(values, targets)

        assert loss.item() > 0

    def test_value_loss_clipping(self):
        """Test value loss clipping."""
        loss_fn = ValueLoss(clip_value_loss=True)

        values = torch.tensor([10.0])
        targets = torch.tensor([0.0])
        old_values = torch.tensor([0.0])

        loss = loss_fn(values, targets, old_values)

        assert loss.item() > 0


class TestEntropyBonus:
    """Tests for entropy bonus."""

    def test_entropy_bonus_basic(self):
        """Test basic entropy bonus computation."""
        entropy_fn = EntropyBonus(coef=0.01)

        log_probs = torch.tensor([-1.0, -2.0, -3.0])
        entropy = entropy_fn(log_probs)

        assert entropy.item() < 0  # Negative because it's added to loss


class TestGAE:
    """Tests for Generalized Advantage Estimation."""

    def test_gae_basic(self):
        """Test basic GAE computation."""
        gae = GAE(gamma=0.99, lam=0.95)

        rewards = torch.tensor([[1.0, 0.0, 1.0]])
        values = torch.tensor([[0.5, 0.5, 0.5]])
        next_values = torch.tensor([[0.5, 0.5, 0.5]])
        masks = torch.tensor([[1.0, 1.0, 0.0]])

        advantages, returns = gae(rewards, values, next_values, masks)

        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape

    def test_gae_normalization(self):
        """Test advantage normalization."""
        gae = GAE(gamma=0.99, lam=0.95)

        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        values = torch.tensor([[0.0, 0.0, 0.0]])
        next_values = torch.zeros_like(values)
        masks = torch.ones_like(rewards)

        advantages, _ = gae(rewards, values, next_values, masks)

        assert abs(advantages.mean().item()) < 0.1  # Should be approximately normalized


class TestGRPOLoss:
    """Tests for GRPO loss."""

    def test_grpo_basic(self):
        """Test basic GRPO loss computation."""
        loss_fn = GRPOLoss(clip_epsilon=0.2, group_size=4)

        log_probs = torch.randn(8, 10)
        old_log_probs = torch.randn(8, 10)
        rewards = torch.randn(8)

        loss, clip_frac = loss_fn(log_probs, old_log_probs, rewards)

        assert loss.item() != 0
        assert 0 <= clip_frac.item() <= 1

    def test_grpo_group_advantage(self):
        """Test group advantage computation."""
        loss_fn = GRPOLoss(clip_epsilon=0.2, group_size=4)

        # Create rewards where first group has high rewards, second has low
        log_probs = torch.randn(8, 10)
        old_log_probs = torch.zeros_like(log_probs)
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])

        loss, clip_frac = loss_fn(log_probs, old_log_probs, rewards)

        assert loss.item() != 0


class TestReplayBuffer:
    """Tests for replay buffer."""

    def test_push_sample(self):
        """Test pushing and sampling."""
        buffer = ReplayBuffer(capacity=100)

        for i in range(10):
            buffer.push(
                observation=i,
                action=i,
                reward=1.0,
                next_observation=i + 1,
                done=False,
            )

        assert len(buffer) == 10

        samples = buffer.sample(5)
        assert len(samples) == 5

    def test_buffer_full(self):
        """Test buffer capacity."""
        buffer = ReplayBuffer(capacity=5)

        for i in range(10):
            buffer.push(i, i, 1.0, i + 1, False)

        assert len(buffer) == 5  # Should be capped at capacity

    def test_clear(self):
        """Test buffer clearing."""
        buffer = ReplayBuffer(capacity=100)
        buffer.push(1, 1, 1.0, 2, False)
        buffer.clear()
        assert len(buffer) == 0


class TestSequenceBuffer:
    """Tests for sequence buffer."""

    def test_add_sequence(self):
        """Test adding sequences."""
        buffer = SequenceBuffer(capacity=10)

        from rl.replay_buffer import SequenceData
        seq = SequenceData(
            observations=[1, 2, 3],
            actions=[0, 1, 2],
            rewards=[1.0, 1.0, 1.0],
        )

        buffer.add_sequence(seq)
        assert len(buffer) == 1

    def test_sample_sequences(self):
        """Test sampling sequences."""
        buffer = SequenceBuffer(capacity=10)

        from rl.replay_buffer import SequenceData
        for i in range(5):
            seq = SequenceData(
                observations=[i],
                actions=[i],
                rewards=[1.0],
            )
            buffer.add_sequence(seq)

        samples = buffer.sample(3, sample_type="sequence")
        assert len(samples) == 3


class TestPriorityReplayBuffer:
    """Tests for priority replay buffer."""

    def test_priority_sampling(self):
        """Test priority-based sampling."""
        buffer = PriorityReplayBuffer(capacity=100)

        for i in range(20):
            buffer.push(i, i, 1.0, i + 1, False)

        # Update priorities for first few samples
        indices = np.array([0, 1, 2])
        priorities = np.array([10.0, 10.0, 10.0])
        buffer.update_priorities(indices, priorities)

        # Sample - high priority items should be more likely
        samples, sampled_indices, weights = buffer.sample(10)

        assert len(samples) == 10
        assert len(sampled_indices) == 10
        assert len(weights) == 10

    def test_priority_update(self):
        """Test priority updates."""
        buffer = PriorityReplayBuffer(capacity=100)

        for i in range(10):
            buffer.push(i, i, 1.0, i + 1, False)

        indices = np.array([0, 1])
        priorities = np.array([5.0, 5.0])
        buffer.update_priorities(indices, priorities)

        assert buffer._priorities[0] > 0
        assert buffer._priorities[1] > 0


class TestPPOTrainer:
    """Tests for PPO trainer."""

    @pytest.fixture
    def dummy_policy(self):
        """Create a dummy policy for testing."""
        class DummyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)

            def forward_for_training(self, obs, actions):
                return {
                    "log_probs": torch.randn_like(actions).float(),
                    "values": torch.randn(obs.shape[0]).float(),
                    "entropy": torch.randn(1).float(),
                }

            def get_value(self, obs):
                return torch.randn(obs.shape[0]).float()

        return DummyPolicy()

    def test_ppo_trainer_init(self, dummy_policy):
        """Test PPO trainer initialization."""
        config = PPOConfig(lr=0.001, epochs=2)
        trainer = PPOTrainer(dummy_policy, config)

        assert trainer is not None
        assert trainer.optimizer is not None

    def test_ppo_trainer_update(self, dummy_policy):
        """Test PPO trainer update."""
        config = PPOConfig(lr=0.001, epochs=2, batch_size=4, mini_batch_size=2)
        trainer = PPOTrainer(dummy_policy, config)

        rollouts = {
            "observations": torch.randn(8, 10),
            "actions": torch.randint(0, 5, (8, 10)),
            "rewards": torch.randn(8, 10),
            "masks": torch.ones(8, 10),
            "log_probs": torch.randn(8, 10),
            "values": torch.randn(8, 10),
        }

        stats = trainer.update(rollouts)

        assert stats.policy_loss is not None
        assert stats.value_loss is not None


class TestGRPOTrainer:
    """Tests for GRPO trainer."""

    @pytest.fixture
    def dummy_policy(self):
        """Create a dummy policy for testing."""
        class DummyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)

            def forward_for_training(self, obs, actions):
                return {
                    "log_probs": torch.randn_like(actions).float(),
                    "values": torch.randn(obs.shape[0]).float(),
                    "entropy": torch.randn(1).float(),
                }

            def get_value(self, obs):
                return torch.randn(obs.shape[0]).float()

        return DummyPolicy()

    def test_grpo_trainer_init(self, dummy_policy):
        """Test GRPO trainer initialization."""
        config = GRPOConfig(lr=0.001, epochs=2, group_size=4)
        trainer = GRPOTrainer(dummy_policy, config)

        assert trainer is not None

    def test_grpo_trainer_update(self, dummy_policy):
        """Test GRPO trainer update."""
        config = GRPOConfig(lr=0.001, epochs=2, batch_size=8, mini_batch_size=4)
        trainer = GRPOTrainer(dummy_policy, config)

        rollouts = {
            "observations": torch.randn(8, 10),
            "actions": torch.randint(0, 5, (8, 10)),
            "rewards": torch.randn(8, 10),
            "masks": torch.ones(8, 10),
            "log_probs": torch.randn(8, 10),
        }

        stats = trainer.update(rollouts)

        assert stats.policy_loss is not None
        assert stats.reward_mean is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
