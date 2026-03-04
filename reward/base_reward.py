"""Base Reward Module.

Defines the abstract base class for reward functions in the RL-LLM Agent framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class RewardInfo:
    """Information about a reward computation.

    Attributes:
        reward: The computed reward value
        components: Breakdown of reward components
        metadata: Additional metadata
    """
    reward: float
    components: Dict[str, float]
    metadata: Dict[str, Any]


class BaseReward(ABC):
    """Abstract base class for reward functions.

    This class defines the standard interface for reward computation
    in the RL-LLM Agent framework. All reward functions should inherit
    from this class and implement the abstract methods.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the reward function.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._name = self.__class__.__name__

    @property
    def name(self) -> str:
        """Get the name of the reward function."""
        return self._name

    @abstractmethod
    def compute(self, **kwargs) -> float:
        """Compute the reward.

        Args:
            **kwargs: Environment-specific arguments for reward computation

        Returns:
            The computed reward value
        """
        pass

    def compute_with_info(self, **kwargs) -> RewardInfo:
        """Compute reward with detailed information.

        Args:
            **kwargs: Environment-specific arguments

        Returns:
            RewardInfo with reward value and breakdown
        """
        reward = self.compute(**kwargs)
        return RewardInfo(
            reward=reward,
            components={"total": reward},
            metadata={},
        )

    def normalize(self, reward: float, min_reward: float = -1.0, max_reward: float = 1.0) -> float:
        """Normalize reward to a specified range.

        Args:
            reward: Raw reward value
            min_reward: Minimum reward for normalization
            max_reward: Maximum reward for normalization

        Returns:
            Normalized reward in range [-1, 1]
        """
        if max_reward == min_reward:
            return 0.0

        # Scale to [-1, 1]
        normalized = 2 * (reward - min_reward) / (max_reward - min_reward) - 1
        return max(-1.0, min(1.0, normalized))

    def clip(self, reward: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
        """Clip reward to specified bounds.

        Args:
            reward: Reward value to clip
            min_val: Minimum value (default from config)
            max_val: Maximum value (default from config)

        Returns:
            Clipped reward value
        """
        if min_val is None:
            min_val = self.config.get("clip_min", -float("inf"))
        if max_val is None:
            max_val = self.config.get("clip_max", float("inf"))
        return max(min_val, min(max_val, reward))

    def reset(self) -> None:
        """Reset any internal state.

        Override this method if the reward function maintains state.
        """
        pass

    def __call__(self, **kwargs) -> float:
        """Allow the reward to be called as a function.

        Args:
            **kwargs: Arguments for reward computation

        Returns:
            Computed reward
        """
        return self.compute(**kwargs)
