"""GUI Navigation Environment.

Provides a simulated environment for GUI interaction tasks.
Supports tasks like web navigation, desktop automation, etc.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from .base_env import (
    BaseEnvironment,
    Observation,
    Action,
    ActionSpace,
    ObservationSpace,
    StepResult,
)


@dataclass
class GUIElement:
    """Represents a GUI element.

    Attributes:
        id: Element identifier
        element_type: Type of element (button, input, text, etc.)
        text: Text content
        bbox: Bounding box [x1, y1, x2, y2]
        attributes: HTML attributes or properties
        children: Child elements
        parent_id: Parent element ID
        enabled: Whether element is interactable
    """
    id: str
    element_type: str
    text: str = ""
    bbox: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    enabled: bool = True


@dataclass
class GUITask:
    """Represents a GUI navigation task.

    Attributes:
        id: Task identifier
        instruction: Task instruction
        start_url: Starting URL or screen
        goal_description: Description of the goal state
        goal_check: Function to check if goal is reached
        elements: Initial DOM/screen elements
    """
    id: str
    instruction: str
    start_url: str
    goal_description: str
    goal_check: Any  # Callable or criteria
    elements: List[GUIElement] = field(default_factory=list)


@dataclass
class GUIState:
    """Represents the current state of the GUI environment.

    Attributes:
        current_url: Current URL or screen identifier
        elements: Current visible elements
        action_history: History of actions taken
        page_history: History of visited pages/screens
        goal_reached: Whether the goal has been reached
    """
    current_url: str = ""
    elements: List[GUIElement] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    page_history: List[str] = field(default_factory=list)
    goal_reached: bool = False


class GUIEnvironment(BaseEnvironment):
    """Environment for GUI navigation tasks.

    This environment allows the agent to:
    - Navigate web pages or GUI screens
    - Interact with elements (click, type, select)
    - Observe the current screen state
    - Complete tasks through GUI interaction

    Actions:
    - 'click': Click on an element
    - 'type': Type text into an input field
    - 'navigate': Navigate to a URL
    - 'scroll': Scroll the page
    - 'select': Select an option from dropdown
    - 'submit': Submit current state as goal reached

    Rewards:
    - Positive reward for reaching goal
    - Small reward for progress indicators
    - Penalty for redundant actions
    - Step penalty for efficiency
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self._current_task: Optional[GUITask] = None
        self._state: GUIState = GUIState()
        self._screen_width = self.config.get("screen_width", 1024)
        self._screen_height = self.config.get("screen_height", 768)
        self._max_elements = self.config.get("max_elements", 100)

        # Element registry
        self._element_registry: Dict[str, GUIElement] = {}

    def load_task(self, task: GUITask) -> None:
        """Load a GUI task.

        Args:
            task: The GUI task to load
        """
        self._current_task = task
        self._state = GUIState(
            current_url=task.start_url,
            elements=task.elements.copy(),
        )
        self._build_element_registry(task.elements)

    def _build_element_registry(self, elements: List[GUIElement]) -> None:
        """Build element registry from element list.

        Args:
            elements: List of GUI elements
        """
        self._element_registry = {}
        for elem in elements:
            self._register_element(elem)

    def _register_element(self, element: GUIElement) -> None:
        """Register an element.

        Args:
            element: GUI element to register
        """
        self._element_registry[element.id] = element
        for child_id in element.children:
            if child_id in self._element_registry:
                self._register_element(self._element_registry[child_id])

    def _get_visible_elements(self) -> List[GUIElement]:
        """Get currently visible elements.

        Returns:
            List of visible GUI elements
        """
        # In a real implementation, this would filter based on viewport
        return list(self._element_registry.values())[:self._max_elements]

    def reset(self, seed: Optional[int] = None, task: Optional[GUITask] = None, **kwargs) -> Observation:
        """Reset the environment.

        Args:
            seed: Random seed
            task: Optional task to load
            **kwargs: Additional arguments

        Returns:
            Initial observation
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._current_step = 0
        self._done = False

        if task is not None:
            self.load_task(task)

        return self.get_observation()

    def step(self, action: Action) -> StepResult:
        """Take an action in the environment.

        Args:
            action: Action with type and value

        Returns:
            StepResult with observation, reward, done, truncated, info
        """
        self._current_step += 1
        reward = self.config.get("step_penalty", -0.01)
        info = {"action_type": action.action_type}
        truncated = self._current_step >= self._max_steps

        try:
            if action.action_type == "click":
                result, click_reward = self._click(action.value)
                reward += click_reward
                info["result"] = result

            elif action.action_type == "type":
                result, type_reward = self._type(action.value)
                reward += type_reward
                info["result"] = result

            elif action.action_type == "navigate":
                result, nav_reward = self._navigate(action.value)
                reward += nav_reward
                info["result"] = result

            elif action.action_type == "scroll":
                result, scroll_reward = self._scroll(action.value)
                reward += scroll_reward
                info["result"] = result

            elif action.action_type == "select":
                result, select_reward = self._select(action.value)
                reward += select_reward
                info["result"] = result

            elif action.action_type == "submit":
                goal_reached, submit_reward = self._check_goal()
                reward += submit_reward
                self._done = True
                info["goal_reached"] = goal_reached

            else:
                reward -= 0.1
                info["error"] = f"Unknown action type: {action.action_type}"

        except Exception as e:
            reward -= 0.5
            info["error"] = str(e)

        # Check goal completion
        if not self._done:
            goal_reached, _ = self._check_goal()
            if goal_reached:
                self._done = True
                reward += self.config.get("success_reward", 1.0)
                info["goal_reached"] = True

        if truncated and not self._done:
            self._done = True
            info["truncation"] = True

        return StepResult(
            observation=self.get_observation(),
            reward=reward,
            done=self._done,
            truncated=truncated,
            info=info,
        )

    def _click(self, element_id: str) -> Tuple[str, float]:
        """Click on an element.

        Args:
            element_id: ID of element to click

        Returns:
            Tuple of (result message, reward)
        """
        if element_id not in self._element_registry:
            return f"Element not found: {element_id}", -0.1

        element = self._element_registry[element_id]

        if not element.enabled:
            return f"Element not clickable: {element_id}", -0.05

        if element.element_type not in ["button", "link", "checkbox", "radio", "option"]:
            return f"Cannot click {element.element_type}: {element_id}", -0.05

        # Record action
        self._state.action_history.append({
            "type": "click",
            "element_id": element_id,
            "step": self._current_step,
        })

        # Simulate click effect (in real implementation, this would update DOM)
        return f"Clicked: {element.text or element_id}", 0.02

    def _type(self, action_data: Dict[str, Any]) -> Tuple[str, float]:
        """Type text into an input field.

        Args:
            action_data: Dictionary with 'element_id' and 'text'

        Returns:
            Tuple of (result message, reward)
        """
        element_id = action_data.get("element_id")
        text = action_data.get("text", "")

        if element_id not in self._element_registry:
            return f"Element not found: {element_id}", -0.1

        element = self._element_registry[element_id]

        if element.element_type not in ["input", "textarea"]:
            return f"Cannot type into {element.element_type}: {element_id}", -0.05

        # Record action
        self._state.action_history.append({
            "type": "type",
            "element_id": element_id,
            "text": text[:50],
            "step": self._current_step,
        })

        return f"Typed into {element_id}", 0.02

    def _navigate(self, url: str) -> Tuple[str, float]:
        """Navigate to a URL.

        Args:
            url: URL to navigate to

        Returns:
            Tuple of (result message, reward)
        """
        # Record action
        self._state.action_history.append({
            "type": "navigate",
            "url": url,
            "step": self._current_step,
        })

        self._state.page_history.append(self._state.current_url)
        self._state.current_url = url

        # In real implementation, this would load new page elements
        return f"Navigated to: {url}", 0.0

    def _scroll(self, direction: str) -> Tuple[str, float]:
        """Scroll the page.

        Args:
            direction: Scroll direction ('up', 'down', 'left', 'right')

        Returns:
            Tuple of (result message, reward)
        """
        if direction not in ["up", "down", "left", "right"]:
            return f"Invalid scroll direction: {direction}", -0.05

        self._state.action_history.append({
            "type": "scroll",
            "direction": direction,
            "step": self._current_step,
        })

        return f"Scrolled {direction}", 0.0

    def _select(self, action_data: Dict[str, Any]) -> Tuple[str, float]:
        """Select an option from a dropdown.

        Args:
            action_data: Dictionary with 'element_id' and 'option'

        Returns:
            Tuple of (result message, reward)
        """
        element_id = action_data.get("element_id")
        option = action_data.get("option")

        if element_id not in self._element_registry:
            return f"Element not found: {element_id}", -0.1

        element = self._element_registry[element_id]

        if element.element_type not in ["select", "dropdown"]:
            return f"Cannot select from {element.element_type}: {element_id}", -0.05

        self._state.action_history.append({
            "type": "select",
            "element_id": element_id,
            "option": option,
            "step": self._current_step,
        })

        return f"Selected {option} from {element_id}", 0.02

    def _check_goal(self) -> Tuple[bool, float]:
        """Check if the goal has been reached.

        Returns:
            Tuple of (goal_reached, reward)
        """
        if self._current_task is None:
            return False, 0.0

        # In a real implementation, this would check actual goal criteria
        # For now, use a simple heuristic
        goal_check = self._current_task.goal_check

        if callable(goal_check):
            try:
                result = goal_check(self._state)
                return bool(result), 1.0 if result else 0.0
            except Exception:
                pass

        # Fallback: check if current state matches goal description
        # This is a simplified heuristic
        return False, 0.0

    def get_observation(self) -> Observation:
        """Get the current observation.

        Returns:
            Current observation
        """
        text = self.get_task_description()
        text += f"\n\nCurrent URL: {self._state.current_url}"

        # Describe visible elements
        elements_text = "\nVisible elements:\n"
        visible = self._get_visible_elements()
        for elem in visible[:20]:  # Limit displayed elements
            elem_desc = f"  [{elem.id}] {elem.element_type}"
            if elem.text:
                elem_desc += f": {elem.text[:30]}"
            if not elem.enabled:
                elem_desc += " (disabled)"
            elements_text += elem_desc + "\n"

        if len(visible) > 20:
            elements_text += f"  ... and {len(visible) - 20} more elements\n"

        text += elements_text
        text += f"\nActions taken: {len(self._state.action_history)}"

        return Observation(
            text=text,
            state={
                "url": self._state.current_url,
                "element_count": len(visible),
                "step": self._current_step,
            },
            metadata={
                "action_history": self._state.action_history[-10:],
                "page_history": self._state.page_history[-5:],
                "goal_reached": self._state.goal_reached,
            },
        )

    def get_action_space(self) -> ActionSpace:
        """Get the action space.

        Returns:
            Action space for GUI environment
        """
        return ActionSpace(
            action_types=["click", "type", "navigate", "scroll", "select", "submit"],
            value_space="structured",
            constraints={
                "max_text_length": 500,
                "valid_scroll_directions": ["up", "down", "left", "right"],
            },
        )

    def get_observation_space(self) -> ObservationSpace:
        """Get the observation space.

        Returns:
            Observation space for GUI environment
        """
        return ObservationSpace(
            text_max_length=8192,
            metadata_keys=["action_history", "page_history", "goal_reached"],
        )

    def compute_reward(self, **kwargs) -> float:
        """Compute reward for current state.

        Args:
            **kwargs: Reward computation arguments

        Returns:
            Computed reward
        """
        reward = 0.0

        # Small reward for exploring new pages
        if len(self._state.page_history) > 0:
            reward += min(len(self._state.page_history) * 0.01, 0.1)

        # Check goal proximity
        if self._state.goal_reached:
            reward += 1.0

        return reward

    def get_info(self) -> Dict[str, Any]:
        """Get additional information.

        Returns:
            Environment information dictionary
        """
        return {
            "current_step": self._current_step,
            "max_steps": self._max_steps,
            "done": self._done,
            "task_id": self._current_task.id if self._current_task else None,
            "current_url": self._state.current_url,
            "actions_taken": len(self._state.action_history),
            "pages_visited": len(self._state.page_history),
            "goal_reached": self._state.goal_reached,
        }

    def is_valid_action(self, action: Action) -> bool:
        """Check if an action is valid.

        Args:
            action: Action to validate

        Returns:
            True if valid, False otherwise
        """
        valid_types = ["click", "type", "navigate", "scroll", "select", "submit"]
        if action.action_type not in valid_types:
            return False

        if action.value is None and action.action_type != "submit":
            return False

        return True

    def get_task_description(self) -> str:
        """Get task description.

        Returns:
            Task description string
        """
        if self._current_task is None:
            return "No task loaded."

        return f"""Task: {self._current_task.id}

Instruction:
{self._current_task.instruction}

Goal:
{self._current_task.goal_description}"""

    def close(self) -> None:
        """Clean up the environment."""
        self._current_task = None
        self._state = GUIState()
        self._element_registry = {}
