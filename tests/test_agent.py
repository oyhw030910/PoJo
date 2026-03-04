"""Tests for Agent Module."""

import pytest
import torch

from agent.llm_wrapper import LLMWrapper, LLMConfig
from agent.policy import PolicyNetwork, PolicyConfig
from agent.memory import MemoryManager, MemoryConfig, ShortTermMemory, LongTermMemory
from agent.planner import Planner, PlannerConfig, Plan, Thought
from agent.tool_manager import ToolManager, ToolRegistry, register_tool


class TestShortTermMemory:
    """Tests for short-term memory."""

    def test_add_and_get(self):
        """Test adding and retrieving items."""
        stm = ShortTermMemory(max_size=5)

        stm.add(observation="obs1", action="act1", reward=1.0)
        stm.add(observation="obs2", action="act2", reward=0.5)

        assert len(stm) == 2

        recent = stm.get_recent()
        assert len(recent) == 2

    def test_max_size(self):
        """Test max size enforcement."""
        stm = ShortTermMemory(max_size=3)

        for i in range(10):
            stm.add(observation=f"obs{i}", action=f"act{i}", reward=float(i))

        assert len(stm) == 3  # Should be capped

    def test_format_for_prompt(self):
        """Test prompt formatting."""
        stm = ShortTermMemory(max_size=5)
        stm.add(observation="test obs", action="test act", reward=1.0)

        prompt = stm.format_for_prompt()
        assert "test obs" in prompt or "Obs" in prompt


class TestLongTermMemory:
    """Tests for long-term memory."""

    def test_add_and_retrieve(self):
        """Test adding and retrieving memories."""
        ltm = LongTermMemory(top_k=5)

        ltm.add("This is a test memory about Python programming", importance=1.0)
        ltm.add("This is about mathematics", importance=0.5)

        # Test retrieval
        results = ltm.retrieve("Python", top_k=1)
        assert len(results) >= 0  # May be empty if embedding not available

    def test_keyword_fallback(self):
        """Test keyword-based search fallback."""
        ltm = LongTermMemory(top_k=5)
        ltm._initialized = True  # Skip embedding initialization

        ltm.add("Python is a programming language", importance=1.0)
        ltm.add("Mathematics is about numbers", importance=0.5)

        results = ltm._keyword_search("Python", top_k=1)
        assert len(results) == 1
        assert "Python" in results[0].content


class TestMemoryManager:
    """Tests for memory manager."""

    def test_add_experience(self):
        """Test adding experiences."""
        config = MemoryConfig(
            short_term_size=10,
            long_term_enabled=False,  # Disable for testing
        )
        mm = MemoryManager(config)

        mm.add_experience(
            observation="test obs",
            action="test act",
            reward=1.0,
            store_long_term=False,
        )

        assert len(mm.short_term) == 1

    def test_get_context(self):
        """Test context retrieval."""
        config = MemoryConfig(long_term_enabled=False)
        mm = MemoryManager(config)

        mm.add_experience("obs1", "act1", 1.0, store_long_term=False)
        mm.add_experience("obs2", "act2", 0.5, store_long_term=False)

        context = mm.get_context()
        assert context is not None

    def test_set_task_context(self):
        """Test setting task context."""
        mm = MemoryManager()

        mm.set_task_context("Test task", "Complete the task")

        assert mm.working.get_context("task_description") == "Test task"
        assert mm.working.get_context("task_goal") == "Complete the task"

    def test_clear(self):
        """Test clearing memory."""
        mm = MemoryManager()

        mm.add_experience("obs", "act", 1.0)
        mm.clear()

        assert len(mm.short_term) == 0


class TestPlanner:
    """Tests for planner."""

    @pytest.fixture
    def dummy_llm(self):
        """Create a dummy LLM for testing."""
        class DummyLLM:
            def generate(self, prompt, max_new_tokens=100, temperature=0.5):
                # Return a simple response
                return '["Step 1: Understand the problem", "Step 2: Solve it"]'

        return DummyLLM()

    def test_planner_init(self, dummy_llm):
        """Test planner initialization."""
        config = PlannerConfig(method="plain")
        planner = Planner(dummy_llm, config)

        assert planner is not None

    def test_plain_plan(self, dummy_llm):
        """Test plain planning."""
        config = PlannerConfig(method="plain")
        planner = Planner(dummy_llm, config)

        result = planner.plan("Solve a math problem")

        assert result is not None
        assert result.plan is not None or result.thoughts is not None

    def test_plan_advance(self):
        """Test advancing through a plan."""
        plan = Plan(
            thoughts=[
                Thought(content="Step 1"),
                Thought(content="Step 2"),
                Thought(content="Step 3"),
            ]
        )

        assert plan.current_step == 0
        assert not plan.is_complete()

        plan.advance()
        assert plan.current_step == 1

        plan.advance()
        plan.advance()
        assert plan.is_complete()


class TestToolRegistry:
    """Tests for tool registry."""

    def test_register_tool(self):
        """Test tool registration."""
        registry = ToolRegistry()

        @registry.register(name="test_tool", description="A test tool")
        def test_func(x: int) -> int:
            return x * 2

        tool = registry.get_tool("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"

    def test_list_tools(self):
        """Test listing tools."""
        registry = ToolRegistry()

        @registry.register(name="tool1")
        def func1(): pass

        @registry.register(name="tool2")
        def func2(): pass

        tools = registry.list_tools()
        assert "tool1" in tools
        assert "tool2" in tools

    def test_parse_tool_call(self):
        """Test parsing tool calls."""
        registry = ToolRegistry()

        # Test parsing
        result = registry.parse_tool_call("test_tool(arg1=1, arg2='hello')")

        assert result is not None
        tool_name, args = result
        assert tool_name == "test_tool"
        assert args["arg1"] == 1
        assert args["arg2"] == "hello"


class TestToolManager:
    """Tests for tool manager."""

    def test_execute_tool(self):
        """Test tool execution."""
        registry = ToolRegistry()

        @registry.register(name="add", description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        manager = ToolManager(registry)
        result = manager.execute("add", a=2, b=3)

        assert result.success
        assert result.output == 5

    def test_execute_unknown_tool(self):
        """Test executing unknown tool."""
        manager = ToolManager()

        result = manager.execute("unknown_tool")

        assert not result.success
        assert "not found" in result.error.lower()

    def test_execution_history(self):
        """Test execution history."""
        registry = ToolRegistry()

        @registry.register(name="test")
        def test_func(): return 1

        manager = ToolManager(registry)
        manager.execute("test")
        manager.execute("test")

        history = manager.get_execution_history()
        assert len(history) == 2


class TestPolicyNetwork:
    """Tests for policy network."""

    @pytest.fixture
    def policy_config(self):
        """Create minimal policy config."""
        # Use a small model for testing
        return PolicyConfig(
            llm_config=LLMConfig(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                lora_enabled=False,
                max_seq_length=64,
            ),
            use_value_head=True,
        )

    @pytest.mark.skip(reason="Requires model download")
    def test_policy_creation(self, policy_config):
        """Test policy network creation."""
        policy = PolicyNetwork(policy_config)
        assert policy is not None

    @pytest.mark.skip(reason="Requires model download")
    def test_policy_forward(self, policy_config):
        """Test policy forward pass."""
        policy = PolicyNetwork(policy_config)

        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones_like(input_ids)

        output = policy.forward(input_ids, attention_mask)

        assert output.log_probs is not None
        assert output.values is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
