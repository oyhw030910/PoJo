"""Base Environment Module.

Defines the abstract base class for all environments in the RL-LLM Agent framework.
Environments follow a Gym-like interface with observations, actions, and rewards.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class Observation:
    """Represents an observation from the environment.

    Attributes:
        text: Text representation of the observation (e.g., problem description)
        state: Current state representation (could be structured data)
        metadata: Additional metadata about the observation
    """
    text: str
    state: Optional[Union[np.ndarray, Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


@dataclass
class Action:
    """Represents an action taken by the agent.

    Attributes:
        action_type: Type of action (e.g., 'text', 'code', 'click')
        value: The actual action value
        metadata: Additional metadata about the action
    """
    action_type: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionSpace:
    """Defines the action space for an environment.

    Attributes:
        action_types: List of allowed action types
        value_space: Description of the value space (e.g., 'text', 'discrete', 'continuous')
        constraints: Any constraints on actions
    """
    action_types: List[str]
    value_space: str = "text"
    constraints: Dict[str, Any] = field(default_factory=dict)

    def contains(self, action: Action) -> bool:
        """Check if an action is valid in this space."""
        return action.action_type in self.action_types


@dataclass
class ObservationSpace:
    """Defines the observation space for an environment.

    Attributes:
        text_max_length: Maximum length of text observations
        state_shape: Shape of state array (if applicable)
        metadata_keys: Expected metadata keys
    """
    text_max_length: int = 4096
    state_shape: Optional[Tuple[int, ...]] = None
    metadata_keys: List[str] = field(default_factory=list)


@dataclass
class StepResult:
    """Result of taking a step in the environment.

    Attributes:
        observation: The resulting observation
        reward: The reward received
        done: Whether the episode is complete
        truncated: Whether the episode was truncated (e.g., max steps)
        info: Additional information
    """
    observation: Observation
    reward: float
    done: bool
    truncated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class BaseEnvironment(ABC):
    """Abstract base class for all environments.

    This class defines the standard interface for environments in the RL-LLM Agent
    framework. All environments should inherit from this class and implement
    the abstract methods.

    The environment follows the standard MDP (Markov Decision Process) interface:
    - reset(): Reset the environment to an initial state
    - step(action): Take an action and return (observation, reward, done, info)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the environment.

        Args:
            config: Configuration dictionary for the environment
        """
        self.config = config or {}
        self._current_step: int = 0
        self._max_steps: int = self.config.get("max_steps", 50)
        self._done: bool = False
        self._rng: np.random.Generator = np.random.default_rng(
            self.config.get("seed", None)
        )

    @property
    def current_step(self) -> int:
        """Get the current step number."""
        return self._current_step

    @property
    def max_steps(self) -> int:
        """Get the maximum number of steps."""
        return self._max_steps

    @property
    def done(self) -> bool:
        """Check if the episode is done."""
        return self._done

    @abstractmethod
    def reset(self, seed: Optional[int] = None, **kwargs) -> Observation:
        """Reset the environment to an initial state.

        Args:
            seed: Optional random seed for reproducibility
            **kwargs: Additional environment-specific arguments

        Returns:
            The initial observation
        """
        pass

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Take an action in the environment.

        Args:
            action: The action to take

        Returns:
            StepResult containing observation, reward, done, truncated, and info
        """
        pass

    @abstractmethod
    def get_observation(self) -> Observation:
        """Get the current observation without taking a step.

        Returns:
            The current observation
        """
        pass

    @abstractmethod
    def get_action_space(self) -> ActionSpace:
        """Get the action space for this environment.

        Returns:
            The action space
        """
        pass

    @abstractmethod
    def get_observation_space(self) -> ObservationSpace:
        """Get the observation space for this environment.

        Returns:
            The observation space
        """
        pass

    @abstractmethod
    def compute_reward(self, **kwargs) -> float:
        """Compute the reward for a given state/action.

        Args:
            **kwargs: Environment-specific reward computation arguments

        Returns:
            The computed reward
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state.

        Returns:
            Dictionary containing environment-specific information
        """
        pass

    @abstractmethod
    def is_valid_action(self, action: Action) -> bool:
        """Check if an action is valid in the current state.

        Args:
            action: The action to validate

        Returns:
            True if the action is valid, False otherwise
        """
        pass

    @abstractmethod
    def get_task_description(self) -> str:
        """Get a description of the current task.

        Returns:
            Task description string
        """
        pass

    def render(self) -> Optional[str]:
        """Render the environment state.

        Returns:
            Optional string representation of the environment
        """
        return None

    def close(self) -> None:
        """Clean up the environment."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
