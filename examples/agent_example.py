"""Example Agent - Comprehensive LLM Agent Example.

This module demonstrates how to create and use a complete LLM Agent
with memory, planning, and tool usage capabilities.
"""

import torch
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from agent.llm_wrapper import LLMWrapper, LLMConfig
from agent.policy import PolicyNetwork, PolicyConfig
from agent.memory import MemoryManager, MemoryConfig
from agent.planner import Planner, PlannerConfig
from agent.tool_manager import ToolManager, ToolRegistry
from environment.base_env import Action, Observation


@dataclass
class AgentConfig:
    """Configuration for the Agent.

    Attributes:
        model_name: HuggingFace model name
        use_memory: Whether to use memory system
        use_planner: Whether to use planning
        device: Device to run on
        max_steps: Maximum steps per episode
    """
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    use_memory: bool = True
    use_planner: bool = True
    device: str = "auto"  # Auto-detect: cuda > mps > cpu
    max_steps: int = 50
    lora_enabled: bool = False


class LLMAgent:
    """A comprehensive LLM Agent with memory, planning, and tools.

    This agent combines:
    - LLM for language understanding and generation
    - Memory for context retention
    - Planner for complex task decomposition
    - Tools for extended capabilities

    Usage:
        config = AgentConfig()
        agent = LLMAgent(config)

        # Run an episode
        result = agent.run(task="Write a function to add two numbers")
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the agent.

        Args:
            config: Agent configuration
        """
        self.config = config or AgentConfig()
        self.device = torch.device(self.config.device)

        # Initialize components
        self._init_llm()
        self._init_memory()
        self._init_planner()
        self._init_tools()

        # State
        self._current_task: Optional[str] = None
        self._step_count: int = 0
        self._episode_history: List[Dict[str, Any]] = []

    def _init_llm(self) -> None:
        """Initialize the LLM."""
        llm_config = LLMConfig(
            model_name=self.config.model_name,
            lora_enabled=self.config.lora_enabled,
            device=self.config.device,
        )
        self.llm = LLMWrapper(llm_config)

        # Create policy network
        policy_config = PolicyConfig(llm_config=llm_config)
        self.policy = PolicyNetwork(policy_config)

        print(f"LLM initialized: {self.config.model_name}")

    def _init_memory(self) -> None:
        """Initialize memory system."""
        if self.config.use_memory:
            memory_config = MemoryConfig(
                short_term_size=10,
                long_term_enabled=True,
                long_term_top_k=3,
            )
            self.memory = MemoryManager(memory_config)
            print("Memory system initialized")
        else:
            self.memory = None

    def _init_planner(self) -> None:
        """Initialize planner."""
        if self.config.use_planner:
            planner_config = PlannerConfig(
                method="react",
                max_iterations=10,
            )
            self.planner = Planner(self.llm, planner_config)
            print("Planner initialized")
        else:
            self.planner = None

    def _init_tools(self) -> None:
        """Initialize tool system."""
        self.tool_manager = ToolManager()

        # Register built-in tools
        self._register_default_tools()
        print("Tool system initialized")

    def _register_default_tools(self) -> None:
        """Register default tools."""
        from tools.code_tool import execute_code, check_code_syntax, analyze_code
        from tools.search_tool import search, calculate

        # Tools are already registered via decorator
        pass

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self._current_task = None
        self._step_count = 0
        self._episode_history = []

        if self.memory:
            self.memory.clear()

    def think(self, context: str) -> str:
        """Generate a thought/response.

        Args:
            context: Current context/prompt

        Returns:
            Generated thought
        """
        response = self.llm.generate(
            prompt=context,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
        )
        return response

    def plan(self, task: str) -> List[str]:
        """Generate a plan for the task.

        Args:
            task: Task description

        Returns:
            List of plan steps
        """
        if self.planner is None:
            return [f"Complete: {task}"]

        result = self.planner.plan(task)

        if result.plan:
            return [t.content for t in result.plan.thoughts]
        return [f"Complete: {task}"]

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool.

        Args:
            tool_name: Name of tool
            **kwargs: Tool arguments

        Returns:
            Tool result
        """
        result = self.tool_manager.execute(tool_name, **kwargs)

        # Store in memory
        if self.memory:
            self.memory.add_experience(
                observation=f"Executed {tool_name}",
                action=str(kwargs),
                reward=0.1 if result.success else -0.1,
            )

        return result

    def run(
        self,
        task: str,
        env: Optional[Any] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run the agent on a task.

        Args:
            task: Task description
            env: Optional environment
            verbose: Whether to print progress

        Returns:
            Episode results
        """
        self.reset()
        self._current_task = task

        # Store task in memory
        if self.memory:
            self.memory.set_task_context(task, "Complete the task")

        if verbose:
            print(f"\n{'='*50}")
            print(f"Task: {task}")
            print(f"{'='*50}")

        # Generate plan
        if self.planner and env is not None:
            if verbose:
                print("\nGenerating plan...")
            plan_result = self.planner.plan(task, env=env)

            if verbose and plan_result.plan:
                print("\nPlan:")
                for i, thought in enumerate(plan_result.plan.thoughts, 1):
                    print(f"  {i}. {thought.content}")

        # Execute in environment
        if env is not None:
            return self._run_in_environment(task, env, verbose)
        else:
            return self._run_conversation(task, verbose)

    def _run_in_environment(
        self,
        task: str,
        env: Any,
        verbose: bool
    ) -> Dict[str, Any]:
        """Run agent in an environment.

        Args:
            task: Task description
            env: Environment
            verbose: Whether to print progress

        Returns:
            Episode results
        """
        obs = env.reset()
        total_reward = 0.0
        done = False

        if verbose:
            print(f"\nInitial observation:\n{obs.text[:200]}...\n")

        while not done and self._step_count < self.config.max_steps:
            self._step_count += 1

            # Get memory context
            context = ""
            if self.memory:
                context = self.memory.get_context(num_recent=3)

            # Build prompt
            prompt = self._build_action_prompt(obs, context, task)

            # Generate action
            action_text = self.think(prompt)
            action = self._parse_action(action_text)

            if verbose:
                print(f"Step {self._step_count}: Action = {action.action_type}")

            # Execute action
            result = env.step(action)
            obs = result.observation
            total_reward += result.reward
            done = result.done

            # Store experience
            if self.memory:
                self.memory.add_experience(
                    observation=obs.text[:100] if hasattr(obs, 'text') else str(obs)[:100],
                    action=str(action.value)[:50] if action.value else "",
                    reward=result.reward,
                )

            if verbose:
                print(f"  Reward: {result.reward:.3f}, Done: {done}")

        if verbose:
            print(f"\n{'='*50}")
            print(f"Episode complete!")
            print(f"Total steps: {self._step_count}")
            print(f"Total reward: {total_reward:.3f}")
            print(f"{'='*50}")

        return {
            "task": task,
            "steps": self._step_count,
            "total_reward": total_reward,
            "success": done,
            "history": self._episode_history.copy(),
        }

    def _run_conversation(
        self,
        task: str,
        verbose: bool
    ) -> Dict[str, Any]:
        """Run agent in conversation mode.

        Args:
            task: Task description
            verbose: Whether to print progress

        Returns:
            Episode results
        """
        # Build conversation prompt
        prompt = f"""You are a helpful AI assistant. Please complete the following task:

Task: {task}

Your response:"""

        response = self.think(prompt)

        if verbose:
            print(f"\nResponse:\n{response}")

        return {
            "task": task,
            "response": response,
            "steps": 1,
        }

    def _build_action_prompt(
        self,
        observation: Observation,
        context: str,
        task: str
    ) -> str:
        """Build prompt for action generation.

        Args:
            observation: Current observation
            context: Memory context
            task: Current task

        Returns:
            Prompt string
        """
        obs_text = observation.text if hasattr(observation, 'text') else str(observation)

        prompt = f"""You are an AI agent completing a task.

Task: {task}

{context if context else ''}

Current observation:
{obs_text[:500]}

What action should you take next?
Format: ACTION: <action_type> VALUE: <action_value>

Response:"""

        return prompt

    def _parse_action(self, text: str) -> Action:
        """Parse action from text.

        Args:
            text: Action text

        Returns:
            Action object
        """
        # Simple parsing
        action_type = "text"
        value = text

        if "ACTION:" in text.upper():
            parts = text.upper().split("ACTION:")
            if len(parts) > 1:
                action_part = parts[1].split("VALUE:")[0].strip()
                action_type = action_part.lower()

        if "VALUE:" in text.upper():
            parts = text.upper().split("VALUE:")
            if len(parts) > 1:
                value = parts[1].strip()

        return Action(action_type=action_type, value=value)

    def get_memory_summary(self) -> str:
        """Get memory summary.

        Returns:
            Memory summary string
        """
        if self.memory is None:
            return "Memory not enabled"

        stats = self.memory.get_stats()
        return f"Short-term: {stats['short_term_size']} items, Long-term: {stats['long_term_size']} items"

    def save(self, path: str) -> None:
        """Save agent model.

        Args:
            path: Save path
        """
        self.policy.save_pretrained(path)
        print(f"Agent saved to {path}")

    def load(self, path: str) -> None:
        """Load agent model.

        Args:
            path: Load path
        """
        # Reload policy weights
        self.policy = PolicyNetwork.from_pretrained(path)
        print(f"Agent loaded from {path}")


def create_example_agent() -> LLMAgent:
    """Create an example agent with default settings.

    Returns:
        Configured LLMAgent
    """
    config = AgentConfig(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        use_memory=True,
        use_planner=True,
        max_steps=30,
    )
    return LLMAgent(config)


# Example usage
if __name__ == "__main__":
    # Create agent
    agent = create_example_agent()

    # Example 1: Simple conversation
    print("\n" + "="*60)
    print("Example 1: Simple Conversation")
    print("="*60)
    agent.run(task="Explain what reinforcement learning is", verbose=True)

    # Example 2: Code task (requires environment)
    # from environment.code_env import CodeEnvironment
    # env = CodeEnvironment()
    # agent.run(task="Write a function to calculate factorial", env=env)
