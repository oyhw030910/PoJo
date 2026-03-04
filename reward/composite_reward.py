"""Composite Reward Module.

Provides composite reward functions that combine multiple reward sources.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from .base_reward import BaseReward, RewardInfo
from .task_reward import TaskReward
from .shaping_reward import ShapingReward


@dataclass
class RewardComponent:
    """A component of the composite reward.

    Attributes:
        name: Component name
        reward_fn: Reward function instance
        weight: Weight for this component
        enabled: Whether this component is enabled
    """
    name: str
    reward_fn: BaseReward
    weight: float = 1.0
    enabled: bool = True


class CompositeReward(BaseReward):
    """Composite reward that combines multiple reward functions.

    This class allows combining multiple reward functions with different
    weights to create a unified reward signal.

    Example:
        composite = CompositeReward()
        composite.add_component("task", TaskReward(), weight=1.0)
        composite.add_component("shaping", ShapingReward(), weight=0.1)
        reward = composite.compute(task_kwargs={...}, shaping_kwargs={...})

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self._components: Dict[str, RewardComponent] = {}
        self._normalize = self.config.get("normalize", False)
        self._clip = self.config.get("clip", True)

    def add_component(
        self,
        name: str,
        reward_fn: BaseReward,
        weight: float = 1.0,
        enabled: bool = True
    ) -> None:
        """Add a reward component.

        Args:
            name: Unique name for this component
            reward_fn: Reward function instance
            weight: Weight for this component's reward
            enabled: Whether this component is enabled
        """
        self._components[name] = RewardComponent(
            name=name,
            reward_fn=reward_fn,
            weight=weight,
            enabled=enabled,
        )

    def remove_component(self, name: str) -> None:
        """Remove a reward component.

        Args:
            name: Name of component to remove
        """
        if name in self._components:
            del self._components[name]

    def enable_component(self, name: str) -> None:
        """Enable a reward component.

        Args:
            name: Name of component to enable
        """
        if name in self._components:
            self._components[name].enabled = True

    def disable_component(self, name: str) -> None:
        """Disable a reward component.

        Args:
            name: Name of component to disable
        """
        if name in self._components:
            self._components[name].enabled = False

    def set_weight(self, name: str, weight: float) -> None:
        """Set the weight of a component.

        Args:
            name: Component name
            weight: New weight value
        """
        if name in self._components:
            self._components[name].weight = weight

    def compute(self, **kwargs) -> float:
        """Compute composite reward.

        Args:
            **kwargs: Keyword arguments passed to component reward functions.
                     Can include:
                     - component_name_kwargs: Dict of kwargs for specific component
                     - Global kwargs passed to all components

        Returns:
            Weighted sum of component rewards
        """
        total_reward = 0.0
        total_weight = 0.0

        for name, component in self._components.items():
            if not component.enabled:
                continue

            # Get component-specific kwargs or use global kwargs
            component_kwargs = kwargs.get(f"{name}_kwargs", kwargs)

            try:
                reward = component.reward_fn.compute(**component_kwargs)
            except Exception:
                reward = 0.0

            if self._normalize:
                reward = component.reward_fn.normalize(reward)

            if self._clip:
                reward = component.reward_fn.clip(reward)

            total_reward += component.weight * reward
            total_weight += component.weight

        if total_weight > 0:
            return total_reward / total_weight
        return 0.0

    def compute_with_info(self, **kwargs) -> RewardInfo:
        """Compute composite reward with detailed breakdown.

        Args:
            **kwargs: Arguments for reward computation

        Returns:
            RewardInfo with per-component breakdown
        """
        components = {}
        total_reward = 0.0
        total_weight = 0.0
        all_metadata = {}

        for name, component in self._components.items():
            if not component.enabled:
                continue

            component_kwargs = kwargs.get(f"{name}_kwargs", kwargs)

            try:
                info = component.reward_fn.compute_with_info(**component_kwargs)
            except Exception:
                info = RewardInfo(reward=0.0, components={}, metadata={})

            reward = info.reward
            if self._normalize:
                reward = component.reward_fn.normalize(reward)
            if self._clip:
                reward = component.reward_fn.clip(reward)

            weighted_reward = component.weight * reward
            components[name] = weighted_reward
            total_reward += weighted_reward
            total_weight += component.weight
            all_metadata[f"{name}_info"] = info

        if total_weight > 0:
            total_reward /= total_weight

        return RewardInfo(
            reward=total_reward,
            components=components,
            metadata={
                "weights": {n: c.weight for n, c in self._components.items() if c.enabled},
                "component_infos": all_metadata,
            },
        )

    def reset(self) -> None:
        """Reset all component rewards."""
        for component in self._components.values():
            component.reward_fn.reset()

    @classmethod
    def create_default(cls, config: Optional[Dict[str, Any]] = None) -> "CompositeReward":
        """Create a composite reward with default components.

        Args:
            config: Configuration dictionary

        Returns:
            CompositeReward with task and shaping rewards
        """
        composite = cls(config)

        # Add task reward
        task_config = config.get("task_reward", {}) if config else {}
        composite.add_component(
            "task",
            TaskReward(task_config),
            weight=config.get("task_weight", 1.0) if config else 1.0,
        )

        # Add shaping reward
        shaping_config = config.get("shaping_reward", {}) if config else {}
        composite.add_component(
            "shaping",
            ShapingReward(shaping_config),
            weight=config.get("shaping_weight", 0.1) if config else 0.1,
        )

        return composite


class HierarchicalReward(BaseReward):
    """Hierarchical reward structure for multi-level tasks.

    Provides rewards at different levels of task hierarchy:
    - Sub-task completion
    - Main task completion
    - Overall episode success
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self._subtask_rewards: Dict[str, BaseReward] = {}
        self._subtask_weights: Dict[str, float] = {}
        self._completed_subtasks: List[str] = []

    def add_subtask(self, subtask_id: str, reward_fn: BaseReward, weight: float = 1.0) -> None:
        """Add a subtask reward.

        Args:
            subtask_id: Unique subtask identifier
            reward_fn: Reward function for this subtask
            weight: Weight for this subtask
        """
        self._subtask_rewards[subtask_id] = reward_fn
        self._subtask_weights[subtask_id] = weight

    def complete_subtask(self, subtask_id: str, **kwargs) -> float:
        """Mark a subtask as complete and compute its reward.

        Args:
            subtask_id: Subtask identifier
            **kwargs: Arguments for reward computation

        Returns:
            Subtask reward
        """
        if subtask_id not in self._subtask_rewards:
            return 0.0

        if subtask_id not in self._completed_subtasks:
            self._completed_subtasks.append(subtask_id)

        return self._subtask_rewards[subtask_id].compute(**kwargs)

    def compute(self, **kwargs) -> float:
        """Compute hierarchical reward.

        Args:
            **kwargs: Reward computation arguments

        Returns:
            Hierarchical reward
        """
        total_reward = 0.0
        total_weight = 0.0

        # Compute subtask rewards
        for subtask_id, reward_fn in self._subtask_rewards.items():
            weight = self._subtask_weights[subtask_id]
            subtask_kwargs = kwargs.get(f"subtask_{subtask_id}", {})

            if subtask_id in self._completed_subtasks:
                reward = reward_fn.compute(**subtask_kwargs)
            else:
                reward = 0.0

            total_reward += weight * reward
            total_weight += weight

        # Add main task reward if provided
        if "main_task_kwargs" in kwargs:
            main_reward = kwargs.get("main_task_reward_fn", lambda **kw: 0.0)
            total_reward += main_reward.compute(**kwargs["main_task_kwargs"])

        if total_weight > 0:
            return total_reward / total_weight
        return 0.0

    def reset(self) -> None:
        """Reset hierarchical reward state."""
        self._completed_subtasks = []
        for reward_fn in self._subtask_rewards.values():
            reward_fn.reset()


class CurriculumReward(BaseReward):
    """Curriculum-based reward that changes over training.

    Adjusts reward weights or thresholds based on training progress.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self._base_reward_fn: Optional[BaseReward] = None
        self._stages: List[Dict[str, Any]] = []
        self._current_stage = 0
        self._training_steps = 0

    def set_base_reward(self, reward_fn: BaseReward) -> None:
        """Set the base reward function.

        Args:
            reward_fn: Base reward function
        """
        self._base_reward_fn = reward_fn

    def add_stage(self, step_threshold: int, weight_modifier: float = 1.0, **kwargs) -> None:
        """Add a training stage.

        Args:
            step_threshold: Training step threshold for this stage
            weight_modifier: Reward weight modifier for this stage
            **kwargs: Additional stage-specific config
        """
        self._stages.append({
            "step_threshold": step_threshold,
            "weight_modifier": weight_modifier,
            **kwargs,
        })
        self._stages.sort(key=lambda x: x["step_threshold"])

    def update_stage(self, training_steps: int) -> None:
        """Update current stage based on training steps.

        Args:
            training_steps: Current training step count
        """
        self._training_steps = training_steps

        for i, stage in enumerate(self._stages):
            if training_steps >= stage["step_threshold"]:
                self._current_stage = i

    def compute(self, **kwargs) -> float:
        """Compute curriculum reward.

        Args:
            **kwargs: Reward computation arguments

        Returns:
            Curriculum-adjusted reward
        """
        if self._base_reward_fn is None:
            return 0.0

        base_reward = self._base_reward_fn.compute(**kwargs)

        # Apply stage modifier
        if self._stages:
            modifier = self._stages[self._current_stage]["weight_modifier"]
            return base_reward * modifier

        return base_reward

    def reset(self) -> None:
        """Reset curriculum state."""
        self._current_stage = 0
        self._training_steps = 0
        if self._base_reward_fn:
            self._base_reward_fn.reset()
