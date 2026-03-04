"""Replay Buffer for RL.

Implements experience replay buffers for storing and sampling transitions.
"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


@dataclass
class Transition:
    """A single transition (experience tuple).

    Attributes:
        observation: Environment observation
        action: Action taken
        reward: Reward received
        next_observation: Next observation
        done: Whether episode ended
        truncated: Whether episode was truncated
        info: Additional information
        action_log_prob: Log probability of action
        value: Value estimate (if available)
    """
    observation: Any
    action: Any
    reward: float
    next_observation: Any
    done: bool
    truncated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    action_log_prob: Optional[float] = None
    value: Optional[float] = None


@dataclass
class SequenceData:
    """Sequence data for trajectory-based learning.

    Attributes:
        observations: List of observations
        actions: List of actions
        rewards: List of rewards
        log_probs: List of log probabilities
        values: List of value estimates
        dones: List of done flags
        masks: List of continuation masks
    """
    observations: List[Any] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    masks: List[float] = field(default_factory=list)

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


class ReplayBuffer:
    """Standard replay buffer for storing transitions.

    Supports uniform sampling and priority sampling.
    """

    def __init__(
        self,
        capacity: int = 10000,
        device: Union[str, torch.device] = "cpu"
    ):
        """Initialize replay buffer.

        Args:
            capacity: Maximum buffer capacity
            device: Device for tensor storage
        """
        self.capacity = capacity
        self.device = device
        self._buffer: deque = deque(maxlen=capacity)
        self._position = 0

    def push(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        done: bool,
        truncated: bool = False,
        **kwargs
    ) -> None:
        """Push a transition to the buffer.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode ended
            truncated: Whether episode was truncated
            **kwargs: Additional transition data
        """
        transition = Transition(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
            truncated=truncated,
            info=kwargs,
        )
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of transitions
        """
        return random.sample(self._buffer, min(batch_size, len(self._buffer)))

    def sample_batch(
        self,
        batch_size: int,
        return_dict: bool = False
    ) -> Union[List[Transition], Dict[str, Any]]:
        """Sample a batch and optionally convert to dict.

        Args:
            batch_size: Number of transitions
            return_dict: Whether to return as dictionary

        Returns:
            List of transitions or dictionary of batched data
        """
        transitions = self.sample(batch_size)

        if not return_dict:
            return transitions

        batch = {
            "observations": [t.observation for t in transitions],
            "actions": [t.action for t in transitions],
            "rewards": [t.reward for t in transitions],
            "next_observations": [t.next_observation for t in transitions],
            "dones": [t.done for t in transitions],
            "truncated": [t.truncated for t in transitions],
        }
        return batch

    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    @property
    def size(self) -> int:
        """Get current size."""
        return len(self._buffer)

    @property
    def full(self) -> bool:
        """Check if buffer is full."""
        return len(self._buffer) >= self.capacity


class SequenceBuffer:
    """Buffer for storing complete sequences/trajectories.

    Used for on-policy algorithms like PPO.
    """

    def __init__(
        self,
        capacity: int = 1000,
        sequence_length: int = 128,
        device: Union[str, torch.device] = "cpu"
    ):
        """Initialize sequence buffer.

        Args:
            capacity: Maximum number of sequences
            sequence_length: Expected sequence length
            device: Device for tensor storage
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.device = device
        self._sequences: List[SequenceData] = []

    def add_sequence(self, sequence: SequenceData) -> None:
        """Add a complete sequence.

        Args:
            sequence: Sequence data to add
        """
        self._sequences.append(sequence)
        if len(self._sequences) > self.capacity:
            self._sequences.pop(0)

    def add_transition(
        self,
        current_sequence: SequenceData,
        observation: Any,
        action: Any,
        reward: float,
        done: bool,
        log_prob: float,
        value: Optional[float] = None,
        truncated: bool = False
    ) -> SequenceData:
        """Add a transition to current sequence.

        Args:
            current_sequence: Current sequence to add to
            observation: Current observation
            action: Action taken
            reward: Reward received
            done: Whether episode ended
            log_prob: Log probability of action
            value: Value estimate
            truncated: Whether truncated

        Returns:
            Updated sequence (or new sequence if episode ended)
        """
        current_sequence.observations.append(observation)
        current_sequence.actions.append(action)
        current_sequence.rewards.append(reward)
        current_sequence.log_probs.append(log_prob)
        if value is not None:
            current_sequence.values.append(value)
        current_sequence.dones.append(done)
        current_sequence.masks.append(0.0 if done or truncated else 1.0)

        if done or truncated:
            self.add_sequence(current_sequence)
            return SequenceData()

        return current_sequence

    def sample(
        self,
        batch_size: int,
        sample_type: str = "sequence"
    ) -> Union[List[SequenceData], List[Transition]]:
        """Sample from the buffer.

        Args:
            batch_size: Number of samples
            sample_type: 'sequence' or 'transition'

        Returns:
            Sampled data
        """
        if sample_type == "sequence":
            return random.sample(
                self._sequences,
                min(batch_size, len(self._sequences))
            )
        else:
            # Sample individual transitions
            transitions = []
            for seq in self._sequences:
                for i in range(len(seq)):
                    transitions.append(Transition(
                        observation=seq.observations[i],
                        action=seq.actions[i],
                        reward=seq.rewards[i],
                        next_observation=seq.observations[i + 1] if i + 1 < len(seq) else None,
                        done=seq.dones[i],
                        info={
                            "log_prob": seq.log_probs[i],
                            "value": seq.values[i] if i < len(seq.values) else None,
                        }
                    ))
            return random.sample(transitions, min(batch_size, len(transitions)))

    def clear(self) -> None:
        """Clear all sequences."""
        self._sequences = []

    def __len__(self) -> int:
        """Get number of sequences."""
        return len(self._sequences)

    @property
    def num_transitions(self) -> int:
        """Get total number of transitions."""
        return sum(len(seq) for seq in self._sequences)


class PriorityReplayBuffer(ReplayBuffer):
    """Priority Experience Replay.

    Samples transitions with probability proportional to TD error.
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        device: Union[str, torch.device] = "cpu"
    ):
        """Initialize priority replay buffer.

        Args:
            capacity: Maximum capacity
            alpha: Priority exponent
            beta: Importance sampling exponent
            beta_increment: Beta increment per sample
            epsilon: Small constant for numerical stability
            device: Device for tensor storage
        """
        super().__init__(capacity, device)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self._priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self._max_priority = 1.0

    def push(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        done: bool,
        truncated: bool = False,
        **kwargs
    ) -> None:
        """Push with max priority.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode ended
            truncated: Whether episode was truncated
            **kwargs: Additional data
        """
        super().push(
            observation, action, reward,
            next_observation, done, truncated, **kwargs
        )

        # Set max priority for new transition
        idx = len(self._buffer) - 1
        if idx < len(self._priorities):
            self._priorities[idx] = self._max_priority ** self.alpha

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Sample with priority.

        Args:
            batch_size: Number of samples

        Returns:
            Tuple of (transitions, indices, importance weights)
        """
        priorities = self._priorities[:len(self._buffer)]
        probs = priorities / (priorities.sum() + self.epsilon)

        indices = np.random.choice(
            len(self._buffer),
            size=min(batch_size, len(self._buffer)),
            p=probs
        )

        transitions = [self._buffer[i] for i in indices]

        # Compute importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (len(self._buffer) * probs[indices]) ** (-self.beta)
        weights = weights / (weights.max() + self.epsilon)

        return transitions, indices, weights

    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray
    ) -> None:
        """Update priorities for sampled transitions.

        Args:
            indices: Indices of sampled transitions
            priorities: New priorities (TD errors)
        """
        for idx, priority in zip(indices, priorities):
            if idx < len(self._priorities):
                self._priorities[idx] = (abs(priority) + self.epsilon) ** self.alpha
                self._max_priority = max(self._max_priority, abs(priority) + self.epsilon)


def collate_transitions(
    transitions: List[Transition],
    device: Union[str, torch.device] = "cpu"
) -> Dict[str, torch.Tensor]:
    """Collate transitions into batch tensors.

    Args:
        transitions: List of transitions
        device: Device for tensors

    Returns:
        Dictionary of batched tensors
    """
    batch = {
        "observations": torch.tensor(
            [t.observation for t in transitions],
            device=device
        ),
        "actions": torch.tensor(
            [t.action for t in transitions],
            device=device
        ),
        "rewards": torch.tensor(
            [t.reward for t in transitions],
            device=device
        ),
        "next_observations": torch.tensor(
            [t.next_observation for t in transitions],
            device=device
        ),
        "dones": torch.tensor(
            [t.done for t in transitions],
            device=device
        ),
    }

    if transitions[0].action_log_prob is not None:
        batch["log_probs"] = torch.tensor(
            [t.action_log_prob for t in transitions],
            device=device
        )

    if transitions[0].value is not None:
        batch["values"] = torch.tensor(
            [t.value for t in transitions],
            device=device
        )

    return batch
