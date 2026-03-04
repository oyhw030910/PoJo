"""Tests for Environment Module."""

import pytest
import torch

from environment.base_env import Observation, Action, ActionSpace, ObservationSpace
from environment.code_env import CodeEnvironment, CodeTask
from environment.math_env import MathEnvironment, MathTask
from environment.gui_env import GUIEnvironment, GUITask, GUIElement


class TestBaseEnvironment:
    """Tests for base environment classes."""

    def test_observation_creation(self):
        """Test observation creation."""
        obs = Observation(text="test observation")
        assert obs.text == "test observation"
        assert str(obs) == "test observation"

    def test_action_creation(self):
        """Test action creation."""
        action = Action(action_type="test", value="test_value")
        assert action.action_type == "test"
        assert action.value == "test_value"

    def test_action_space_contains(self):
        """Test action space validation."""
        space = ActionSpace(action_types=["type1", "type2"])
        valid_action = Action(action_type="type1", value="v")
        invalid_action = Action(action_type="type3", value="v")

        assert space.contains(valid_action)
        assert not space.contains(invalid_action)


class TestCodeEnvironment:
    """Tests for code environment."""

    @pytest.fixture
    def env(self):
        """Create code environment."""
        return CodeEnvironment({"max_steps": 10})

    @pytest.fixture
    def sample_task(self):
        """Create sample code task."""
        return CodeTask(
            id="test_add",
            description="Add two numbers",
            starter_code="def add(a, b):\n    ",
            signature="def add(a, b)",
            test_cases=[
                {"args": [1, 2], "expected": 3},
                {"args": [5, 7], "expected": 12},
            ]
        )

    def test_reset(self, env, sample_task):
        """Test environment reset."""
        obs = env.reset(task=sample_task)
        assert obs is not None
        assert "test_add" in obs.text

    def test_step_generate(self, env, sample_task):
        """Test code generation action."""
        env.reset(task=sample_task)

        action = Action(action_type="generate", value="def add(a, b):\n    return a + b")
        result = env.step(action)

        assert result.reward >= -0.1  # Should not be heavily penalized
        assert "Code generated" in result.info.get("output", "")

    def test_step_execute(self, env, sample_task):
        """Test code execution action."""
        env.reset(task=sample_task)

        # First generate code
        env.step(Action(action_type="generate", value="def add(a, b):\n    return a + b"))

        # Then execute
        action = Action(action_type="execute", value=None)
        result = env.step(action)

        assert result.info.get("output") is not None

    def test_step_submit(self, env, sample_task):
        """Test solution submission."""
        env.reset(task=sample_task)

        # Generate correct solution
        env.step(Action(action_type="generate", value="def add(a, b):\n    return a + b"))

        # Submit
        action = Action(action_type="submit", value=None)
        result = env.step(action)

        assert result.done
        assert result.reward > 0  # Should get positive reward for correct solution

    def test_action_space(self, env):
        """Test action space."""
        space = env.get_action_space()
        assert "generate" in space.action_types
        assert "execute" in space.action_types
        assert "test" in space.action_types
        assert "submit" in space.action_types

    def test_is_valid_action(self, env):
        """Test action validation."""
        valid_action = Action(action_type="generate", value="code")
        invalid_action = Action(action_type="invalid", value="code")
        empty_action = Action(action_type="generate", value=None)

        assert env.is_valid_action(valid_action)
        assert not env.is_valid_action(invalid_action)
        assert not env.is_valid_action(empty_action)


class TestMathEnvironment:
    """Tests for math environment."""

    @pytest.fixture
    def env(self):
        """Create math environment."""
        return MathEnvironment({"max_steps": 10, "allow_calculator": True})

    @pytest.fixture
    def sample_task(self):
        """Create sample math task."""
        return MathTask(
            id="test_math",
            problem="What is 2 + 2?",
            solution="2 + 2 = 4",
            answer="4",
            topic="arithmetic",
        )

    def test_reset(self, env, sample_task):
        """Test environment reset."""
        obs = env.reset(task=sample_task)
        assert obs is not None
        assert "2 + 2" in obs.text

    def test_reason_action(self, env, sample_task):
        """Test reasoning action."""
        env.reset(task=sample_task)

        action = Action(action_type="reason", value="We need to add 2 and 2")
        result = env.step(action)

        assert result.reward >= -0.01

    def test_calculate_action(self, env, sample_task):
        """Test calculator action."""
        env.reset(task=sample_task)

        action = Action(action_type="calculate", value="2 + 2")
        result = env.step(action)

        assert "4" in result.info.get("result", "")

    def test_submit_correct(self, env, sample_task):
        """Test submitting correct answer."""
        env.reset(task=sample_task)

        action = Action(action_type="submit", value="4")
        result = env.step(action)

        assert result.done
        assert result.info.get("correct")
        assert result.reward > 0

    def test_submit_incorrect(self, env, sample_task):
        """Test submitting incorrect answer."""
        env.reset(task=sample_task)

        action = Action(action_type="submit", value="5")
        result = env.step(action)

        assert result.done
        assert not result.info.get("correct")
        assert result.reward < 0

    def test_answer_comparison(self, env):
        """Test answer comparison."""
        assert env._compare_answers("4", "4")
        assert env._compare_answers("4.0", "4")
        assert env._compare_answers("0.5", "1/2")
        assert not env._compare_answers("4", "5")


class TestGUIEnvironment:
    """Tests for GUI environment."""

    @pytest.fixture
    def env(self):
        """Create GUI environment."""
        return GUIEnvironment({"max_steps": 10})

    @pytest.fixture
    def sample_task(self):
        """Create sample GUI task."""
        elements = [
            GUIElement(id="btn1", element_type="button", text="Submit", enabled=True),
            GUIElement(id="input1", element_type="input", text="", enabled=True),
            GUIElement(id="link1", element_type="link", text="Click here", enabled=True),
        ]
        return GUITask(
            id="test_gui",
            instruction="Click the submit button",
            start_url="http://example.com",
            goal_description="Submit button clicked",
            goal_check=lambda state: True,
            elements=elements,
        )

    def test_reset(self, env, sample_task):
        """Test environment reset."""
        obs = env.reset(task=sample_task)
        assert obs is not None
        assert "test_gui" in obs.text

    def test_click_action(self, env, sample_task):
        """Test click action."""
        env.reset(task=sample_task)

        action = Action(action_type="click", value="btn1")
        result = env.step(action)

        assert "Clicked" in result.info.get("result", "")

    def test_type_action(self, env, sample_task):
        """Test type action."""
        env.reset(task=sample_task)

        action = Action(action_type="type", value={"element_id": "input1", "text": "hello"})
        result = env.step(action)

        assert "Typed" in result.info.get("result", "")

    def test_navigate_action(self, env, sample_task):
        """Test navigate action."""
        env.reset(task=sample_task)

        action = Action(action_type="navigate", value="http://other.com")
        result = env.step(action)

        assert "Navigated" in result.info.get("result", "")

    def test_action_space(self, env):
        """Test action space."""
        space = env.get_action_space()
        assert "click" in space.action_types
        assert "type" in space.action_types
        assert "navigate" in space.action_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
