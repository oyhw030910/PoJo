"""Planner Module.

Implements planning strategies for LLM agents including task decomposition,
Tree of Thoughts, Reflection, and ReAct.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy


class PlanningMethod(Enum):
    """Planning method types."""
    REACT = "react"
    TOT = "tot"
    REFLECTION = "reflection"
    DECOMPOSITION = "decomposition"
    PLAIN = "plain"


@dataclass
class PlannerConfig:
    """Configuration for planner.

    Attributes:
        method: Planning method to use
        max_iterations: Maximum planning iterations
        tot_branching_factor: Branching factor for ToT
        tot_depth: Maximum depth for ToT
        reflection_threshold: Reward threshold for triggering reflection
        enable_decomposition: Whether to enable task decomposition
    """
    method: str = "react"
    max_iterations: int = 10
    tot_branching_factor: int = 3
    tot_depth: int = 2
    reflection_threshold: float = -0.5
    enable_decomposition: bool = True


@dataclass
class Thought:
    """A single thought in the planning process.

    Attributes:
        content: Thought content
        action: Optional action to take
        observation: Optional observation from action
        reward: Optional reward received
    """
    content: str
    action: Optional[str] = None
    observation: Optional[str] = None
    reward: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """A plan consisting of multiple thoughts.

    Attributes:
        thoughts: List of thoughts
        status: Plan status (pending, executing, completed, failed)
        current_step: Current step index
    """
    thoughts: List[Thought] = field(default_factory=list)
    status: str = "pending"
    current_step: int = 0

    def add_thought(self, thought: Thought) -> None:
        """Add a thought to the plan."""
        self.thoughts.append(thought)

    def get_current_thought(self) -> Optional[Thought]:
        """Get the current thought."""
        if self.current_step < len(self.thoughts):
            return self.thoughts[self.current_step]
        return None

    def advance(self) -> bool:
        """Advance to next step.

        Returns:
            True if more steps remain, False if plan complete
        """
        self.current_step += 1
        return self.current_step < len(self.thoughts)

    def is_complete(self) -> bool:
        """Check if plan is complete."""
        return self.current_step >= len(self.thoughts)


@dataclass
class PlanningResult:
    """Result of planning process.

    Attributes:
        plan: Generated plan
        thoughts: All thoughts generated
        iterations: Number of iterations used
        success: Whether planning was successful
    """
    plan: Optional[Plan] = None
    thoughts: List[Thought] = field(default_factory=list)
    iterations: int = 0
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskDecomposition:
    """Task decomposition for complex tasks.

    Breaks down complex tasks into manageable subtasks.
    """

    def __init__(self, llm_wrapper: Any):
        """Initialize task decomposition.

        Args:
            llm_wrapper: LLM wrapper for generation
        """
        self.llm = llm_wrapper

    def decompose(
        self,
        task: str,
        max_subtasks: int = 5
    ) -> List[str]:
        """Decompose a task into subtasks.

        Args:
            task: Task description
            max_subtasks: Maximum number of subtasks

        Returns:
            List of subtask descriptions
        """
        prompt = f"""Decompose the following task into {max_subtasks} or fewer subtasks.
Each subtask should be specific and actionable.

Task: {task}

Subtasks (format as JSON array):
"""

        try:
            response = self.llm.generate(
                prompt,
                max_new_tokens=500,
                temperature=0.3,
                do_sample=True,
            )

            # Parse JSON response
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            if start_idx >= 0 and end_idx > start_idx:
                subtasks = json.loads(response[start_idx:end_idx])
                return subtasks[:max_subtasks]
        except Exception:
            pass

        # Fallback: return original task
        return [task]

    def get_dependencies(
        self,
        subtasks: List[str]
    ) -> Dict[int, List[int]]:
        """Get dependencies between subtasks.

        Args:
            subtasks: List of subtasks

        Returns:
            Dictionary mapping subtask index to dependencies
        """
        # Simple heuristic: earlier tasks may be dependencies
        dependencies = {}
        for i in range(len(subtasks)):
            if i > 0:
                dependencies[i] = list(range(i))
            else:
                dependencies[i] = []
        return dependencies


class TreeOfThoughts:
    """Tree of Thoughts planning.

    Explores multiple reasoning paths and selects the best one.
    """

    def __init__(
        self,
        llm_wrapper: Any,
        branching_factor: int = 3,
        max_depth: int = 2,
        evaluator: Optional[Callable[[str], float]] = None
    ):
        """Initialize Tree of Thoughts.

        Args:
            llm_wrapper: LLM wrapper for generation
            branching_factor: Number of branches per node
            max_depth: Maximum tree depth
            evaluator: Optional function to evaluate thoughts
        """
        self.llm = llm_wrapper
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.evaluator = evaluator or self._default_evaluator

    def _default_evaluator(self, thought: str) -> float:
        """Default thought evaluator.

        Args:
            thought: Thought to evaluate

        Returns:
            Score between 0 and 1
        """
        # Simple heuristic: longer, more structured thoughts score higher
        score = min(1.0, len(thought) / 200)
        if "therefore" in thought.lower():
            score += 0.2
        if "because" in thought.lower():
            score += 0.1
        return min(1.0, score)

    def search(
        self,
        problem: str,
        initial_thought: Optional[str] = None
    ) -> List[Thought]:
        """Search for best thought sequence.

        Args:
            problem: Problem description
            initial_thought: Optional initial thought

        Returns:
            Best sequence of thoughts
        """
        # Build tree
        tree = self._build_tree(problem, initial_thought)

        # Evaluate paths
        best_path = self._evaluate_paths(tree)

        return best_path

    def _build_tree(
        self,
        problem: str,
        initial_thought: Optional[str] = None,
        depth: int = 0
    ) -> Dict[str, Any]:
        """Build tree of thoughts.

        Args:
            problem: Problem description
            initial_thought: Initial thought
            depth: Current depth

        Returns:
            Tree structure
        """
        node = {
            "thought": initial_thought or "",
            "children": [],
            "depth": depth,
        }

        if depth >= self.max_depth:
            return node

        # Generate branches
        prompt = f"""Problem: {problem}
Current thought: {initial_thought or 'None'}

Generate {self.branching_factor} different possible next thoughts.
Format as JSON array of strings.
"""

        try:
            response = self.llm.generate(prompt, max_new_tokens=800, temperature=0.7)

            # Parse JSON
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            if start_idx >= 0 and end_idx > start_idx:
                branches = json.loads(response[start_idx:end_idx])

                for branch in branches[:self.branching_factor]:
                    child = self._build_tree(problem, branch, depth + 1)
                    node["children"].append(child)
        except Exception:
            pass

        return node

    def _evaluate_paths(
        self,
        tree: Dict[str, Any],
        path: Optional[List[str]] = None
    ) -> List[Thought]:
        """Evaluate all paths and return best.

        Args:
            tree: Tree structure
            path: Current path

        Returns:
            Best path as list of Thoughts
        """
        if path is None:
            path = []

        current_thought = tree.get("thought", "")
        if current_thought:
            path = path + [current_thought]

        children = tree.get("children", [])
        if not children:
            # Leaf node: evaluate path
            score = sum(self.evaluator(t) for t in path)
            return path, score

        # Recursively evaluate children
        best_child_path = None
        best_score = -float("inf")

        for child in children:
            child_path, child_score = self._evaluate_paths(child, path)
            if child_score > best_score:
                best_score = child_score
                best_child_path = child_path

        return best_child_path or path

    def format_thoughts(self, thoughts: List[str]) -> List[Thought]:
        """Format thoughts as Thought objects.

        Args:
            thoughts: List of thought strings

        Returns:
            List of Thought objects
        """
        return [Thought(content=t) for t in thoughts]


class Reflection:
    """Reflection mechanism for self-improvement.

    Analyzes past performance to improve future decisions.
    """

    def __init__(self, llm_wrapper: Any):
        """Initialize reflection.

        Args:
            llm_wrapper: LLM wrapper for generation
        """
        self.llm = llm_wrapper

    def reflect(
        self,
        trajectory: List[Dict[str, Any]],
        final_reward: float
    ) -> Dict[str, Any]:
        """Reflect on a trajectory.

        Args:
            trajectory: List of (observation, action, reward) tuples
            final_reward: Final reward received

        Returns:
            Reflection results
        """
        # Format trajectory
        traj_str = self._format_trajectory(trajectory)

        prompt = f"""Analyze the following trajectory and provide insights for improvement.

{traj_str}

Final reward: {final_reward}

Provide:
1. What went well
2. What went wrong
3. Suggestions for improvement

Format as JSON.
"""

        try:
            response = self.llm.generate(prompt, max_new_tokens=500, temperature=0.3)

            # Parse JSON
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                return json.loads(response[start_idx:end_idx])
        except Exception:
            pass

        return {
            "went_well": ["Completed the task"],
            "went_wrong": ["Could be more efficient"],
            "suggestions": ["Think more carefully before acting"],
        }

    def _format_trajectory(
        self,
        trajectory: List[Dict[str, Any]]
    ) -> str:
        """Format trajectory as string.

        Args:
            trajectory: Trajectory data

        Returns:
            Formatted string
        """
        lines = ["Trajectory:"]
        for i, step in enumerate(trajectory[:10]):  # Limit to 10 steps
            obs = step.get("observation", "")[:50]
            act = step.get("action", "")[:30]
            rew = step.get("reward", 0)
            lines.append(f"Step {i+1}: Obs={obs}... | Act={act}... | Reward={rew:.2f}")
        return "\n".join(lines)

    def get_improvement_prompt(
        self,
        reflection: Dict[str, Any],
        original_plan: Optional[Plan] = None
    ) -> str:
        """Get prompt for improved planning based on reflection.

        Args:
            reflection: Reflection results
            original_plan: Original plan if available

        Returns:
            Improvement prompt
        """
        suggestions = reflection.get("suggestions", [])
        went_wrong = reflection.get("went_wrong", [])

        prompt = "Based on previous experience:\n"
        if went_wrong:
            prompt += "Issues to avoid:\n"
            for issue in went_wrong:
                prompt += f"  - {issue}\n"
        if suggestions:
            prompt += "Suggestions:\n"
            for sug in suggestions:
                prompt += f"  - {sug}\n"

        return prompt


class ReAct:
    """ReAct (Reason + Act) planning.

    Alternates between reasoning and taking actions.
    """

    def __init__(
        self,
        llm_wrapper: Any,
        max_iterations: int = 10,
        action_space: Optional[List[str]] = None
    ):
        """Initialize ReAct.

        Args:
            llm_wrapper: LLM wrapper for generation
            max_iterations: Maximum iterations
            action_space: List of available actions
        """
        self.llm = llm_wrapper
        self.max_iterations = max_iterations
        self.action_space = action_space or []

    def run(
        self,
        task: str,
        env: Any,
        initial_observation: Optional[str] = None
    ) -> PlanningResult:
        """Run ReAct loop.

        Args:
            task: Task description
            env: Environment
            initial_observation: Initial observation

        Returns:
            Planning result
        """
        thoughts = []
        obs = initial_observation or env.get_task_description()

        for iteration in range(self.max_iterations):
            # Reason
            thought = self._reason(task, obs, thoughts)
            thoughts.append(thought)

            # Check if we should terminate
            if self._should_terminate(thought):
                break

            # Act
            if thought.action:
                action = self._parse_action(thought.action)
                obs, reward, done, _ = self._execute_action(env, action)
                thought.observation = obs
                thought.reward = reward

                if done:
                    break

        # Create plan from thoughts
        plan = Plan(thoughts=thoughts, status="completed")

        return PlanningResult(
            plan=plan,
            thoughts=thoughts,
            iterations=len(thoughts),
            success=done if 'done' in dir() else False,
        )

    def _reason(
        self,
        task: str,
        observation: str,
        thoughts: List[Thought]
    ) -> Thought:
        """Generate next thought.

        Args:
            task: Task description
            observation: Current observation
            thoughts: Previous thoughts

        Returns:
            New thought
        """
        # Build context
        context = f"Task: {task}\n\nObservation: {observation}\n\n"

        if thoughts:
            context += "Previous thoughts:\n"
            for i, t in enumerate(thoughts[-3:]):  # Last 3 thoughts
                context += f"{i+1}. {t.content}\n"
                if t.action:
                    context += f"   Action: {t.action}\n"
                if t.observation:
                    context += f"   Result: {t.observation}\n"

        # Generate thought
        prompt = f"""{context}
What should I do next?

Format:
Thought: [your reasoning]
Action: [action_name] [action_parameter]
"""

        response = self.llm.generate(prompt, max_new_tokens=200, temperature=0.5)

        # Parse response
        thought = Thought(content=response)

        if "Action:" in response:
            action_part = response.split("Action:")[1].strip().split("\n")[0]
            thought.action = action_part

        return thought

    def _should_terminate(self, thought: Thought) -> bool:
        """Check if we should terminate.

        Args:
            thought: Current thought

        Returns:
            Whether to terminate
        """
        content = thought.content.lower()
        return any(term in content for term in [
            "final answer",
            "task complete",
            "submit",
            "done",
        ])

    def _parse_action(self, action_str: str) -> Tuple[str, Any]:
        """Parse action string.

        Args:
            action_str: Action string

        Returns:
            Tuple of (action_type, action_value)
        """
        parts = action_str.strip().split(maxsplit=1)
        if len(parts) == 1:
            return parts[0], None
        return parts[0], parts[1]

    def _execute_action(
        self,
        env: Any,
        action: Tuple[str, Any]
    ) -> Tuple[str, float, bool, Dict]:
        """Execute action in environment.

        Args:
            env: Environment
            action: Action tuple

        Returns:
            Tuple of (observation, reward, done, info)
        """
        from environment.base_env import Action

        env_action = Action(
            action_type=action[0],
            value=action[1],
        )

        result = env.step(env_action)

        return (
            result.observation.text if hasattr(result.observation, 'text') else str(result.observation),
            result.reward,
            result.done,
            result.info,
        )


class Planner:
    """Main planner that coordinates planning strategies.

    Provides unified interface for all planning methods.
    """

    def __init__(
        self,
        llm_wrapper: Any,
        config: Optional[PlannerConfig] = None
    ):
        """Initialize planner.

        Args:
            llm_wrapper: LLM wrapper for generation
            config: Planner configuration
        """
        self.llm = llm_wrapper
        self.config = config or PlannerConfig()

        # Initialize planning components
        self.decomposition = TaskDecomposition(llm_wrapper)
        self.tot = TreeOfThoughts(
            llm_wrapper,
            branching_factor=self.config.tot_branching_factor,
            max_depth=self.config.tot_depth,
        )
        self.reflection = Reflection(llm_wrapper)
        self.react = ReAct(
            llm_wrapper,
            max_iterations=self.config.max_iterations,
        )

        # Current plan
        self._current_plan: Optional[Plan] = None

    def plan(
        self,
        task: str,
        method: Optional[str] = None,
        env: Optional[Any] = None,
        initial_observation: Optional[str] = None
    ) -> PlanningResult:
        """Generate a plan for the task.

        Args:
            task: Task description
            method: Planning method (overrides config)
            env: Optional environment for ReAct
            initial_observation: Initial observation

        Returns:
            Planning result
        """
        method = method or self.config.method

        if method == "react" and env is not None:
            return self.react.run(task, env, initial_observation)

        elif method == "tot":
            thoughts = self.tot.search(task)
            formatted = self.tot.format_thoughts(thoughts)
            plan = Plan(thoughts=formatted)
            return PlanningResult(
                plan=plan,
                thoughts=formatted,
                iterations=len(thoughts),
                success=True,
            )

        elif method == "decomposition":
            subtasks = self.decomposition.decompose(task)
            thoughts = [Thought(content=t) for t in subtasks]
            plan = Plan(thoughts=thoughts)
            return PlanningResult(
                plan=plan,
                thoughts=thoughts,
                iterations=len(subtasks),
                success=True,
            )

        else:
            # Plain planning: simple generation
            return self._plain_plan(task)

    def _plain_plan(self, task: str) -> PlanningResult:
        """Simple planning without complex strategies.

        Args:
            task: Task description

        Returns:
            Planning result
        """
        prompt = f"""Plan how to solve the following task.

Task: {task}

Provide a step-by-step plan.
Format as JSON array of steps.
"""

        try:
            response = self.llm.generate(prompt, max_new_tokens=500, temperature=0.5)

            # Try to parse as JSON
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            if start_idx >= 0 and end_idx > start_idx:
                steps = json.loads(response[start_idx:end_idx])
                thoughts = [Thought(content=s) for s in steps]
            else:
                thoughts = [Thought(content=response)]

            plan = Plan(thoughts=thoughts)

            return PlanningResult(
                plan=plan,
                thoughts=thoughts,
                iterations=1,
                success=True,
            )
        except Exception as e:
            return PlanningResult(
                plan=None,
                thoughts=[],
                iterations=0,
                success=False,
                metadata={"error": str(e)},
            )

    def get_next_action(self, plan: Plan, context: str) -> Optional[str]:
        """Get the next action from a plan.

        Args:
            plan: Current plan
            context: Current context

        Returns:
            Next action or None
        """
        current_thought = plan.get_current_thought()
        if current_thought and current_thought.action:
            return current_thought.action
        return None

    def advance_plan(self, plan: Plan) -> bool:
        """Advance the plan to the next step.

        Args:
            plan: Plan to advance

        Returns:
            True if more steps remain
        """
        return plan.advance()

    def reflect_and_update(
        self,
        plan: Plan,
        trajectory: List[Dict[str, Any]],
        final_reward: float
    ) -> Plan:
        """Reflect on plan execution and update.

        Args:
            plan: Executed plan
            trajectory: Execution trajectory
            final_reward: Final reward

        Returns:
            Updated plan
        """
        # Reflect
        reflection = self.reflection.reflect(trajectory, final_reward)

        # Get improvement suggestions
        if final_reward < self.config.reflection_threshold:
            improvement_prompt = self.reflection.get_improvement_prompt(
                reflection, plan
            )

            # Generate new plan with improvements
            new_task = f"{improvement_prompt}\n\nOriginal task plan: {plan.thoughts}"
            new_result = self.plan(new_task)

            if new_result.success and new_result.plan:
                return new_result.plan

        return plan

    def reset(self) -> None:
        """Reset planner state."""
        self._current_plan = None
