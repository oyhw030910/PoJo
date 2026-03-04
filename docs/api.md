# API Documentation

## Core APIs

### Environment API

#### BaseEnvironment

```python
class BaseEnvironment(ABC):
    """Abstract base class for all environments."""

    def reset(self, seed: Optional[int] = None, **kwargs) -> Observation:
        """Reset the environment to an initial state."""

    def step(self, action: Action) -> StepResult:
        """Take an action and return result."""

    def get_observation(self) -> Observation:
        """Get current observation."""

    def get_action_space(self) -> ActionSpace:
        """Get action space definition."""

    def get_observation_space(self) -> ObservationSpace:
        """Get observation space definition."""

    def compute_reward(self, **kwargs) -> float:
        """Compute reward for current state."""

    def is_valid_action(self, action: Action) -> bool:
        """Check if action is valid."""

    def get_task_description(self) -> str:
        """Get task description."""
```

#### Observation

```python
@dataclass
class Observation:
    text: str                                    # Text representation
    state: Optional[Union[np.ndarray, Dict]]     # Structured state
    metadata: Dict[str, Any]                     # Additional info
```

#### Action

```python
@dataclass
class Action:
    action_type: str                             # Type of action
    value: Any                                   # Action value
    metadata: Dict[str, Any]                     # Additional info
```

#### StepResult

```python
@dataclass
class StepResult:
    observation: Observation                     # Resulting observation
    reward: float                                # Reward received
    done: bool                                   # Episode complete
    truncated: bool                              # Truncated
    info: Dict[str, Any]                         # Additional info
```

### Agent API

#### PolicyNetwork

```python
class PolicyNetwork(nn.Module):
    """Policy network for RL with LLM backbone."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        action_ids: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        """Forward pass through policy."""

    def forward_for_training(
        self,
        observations: Union[torch.Tensor, Dict],
        actions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""

    def get_action(
        self,
        observations: Union[torch.Tensor, Dict],
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action from policy."""

    def get_action_with_log_prob(
        self,
        observations: Union[torch.Tensor, Dict],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and log probability."""

    def get_value(
        self,
        observations: Union[torch.Tensor, Dict],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get value estimate."""
```

#### MemoryManager

```python
class MemoryManager:
    """Main memory manager."""

    def add_experience(
        self,
        observation: str,
        action: str,
        reward: float,
        store_long_term: bool = True,
        importance: Optional[float] = None
    ) -> None:
        """Add an experience to memory."""

    def get_context(
        self,
        query: Optional[str] = None,
        include_short_term: bool = True,
        include_long_term: bool = True,
        num_recent: Optional[int] = None
    ) -> str:
        """Get memory context for decision making."""

    def set_task_context(self, task_description: str, task_goal: str) -> None:
        """Set current task context."""

    def add_note_to_context(self, note: str) -> None:
        """Add a note to working memory."""

    def clear(self) -> None:
        """Clear all memories."""
```

#### Planner

```python
class Planner:
    """Main planner for planning strategies."""

    def plan(
        self,
        task: str,
        method: Optional[str] = None,
        env: Optional[Any] = None,
        initial_observation: Optional[str] = None
    ) -> PlanningResult:
        """Generate a plan for the task."""

    def advance_plan(self, plan: Plan) -> bool:
        """Advance the plan to the next step."""

    def reflect_and_update(
        self,
        plan: Plan,
        trajectory: List[Dict[str, Any]],
        final_reward: float
    ) -> Plan:
        """Reflect on plan execution and update."""

    def reset(self) -> None:
        """Reset planner state."""
```

### RL API

#### PPOTrainer

```python
class PPOTrainer:
    """PPO Trainer for RL optimization."""

    def update(
        self,
        rollouts: Dict[str, torch.Tensor],
        verbose: bool = False
    ) -> PPOStats:
        """Update policy using PPO."""

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        next_values: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""

    def save_checkpoint(self, path: str) -> None:
        """Save trainer checkpoint."""

    def load_checkpoint(self, path: str) -> None:
        """Load trainer checkpoint."""
```

#### GRPOTrainer

```python
class GRPOTrainer:
    """GRPO Trainer for RL optimization."""

    def update(
        self,
        rollouts: Dict[str, torch.Tensor],
        verbose: bool = False
    ) -> GRPOStats:
        """Update policy using GRPO."""

    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
        group_size: Optional[int] = None
    ) -> torch.Tensor:
        """Compute group-relative advantages."""
```

#### RLTrainer

```python
class RLTrainer:
    """Main RL Trainer."""

    def collect_rollout(
        self,
        num_steps: Optional[int] = None,
        deterministic: bool = False
    ) -> RolloutData:
        """Collect rollout data."""

    def update_policy(self, rollout: RolloutData) -> Union[PPOStats, GRPOStats]:
        """Update policy using collected rollout."""

    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """Evaluate the current policy."""

    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        verbose: bool = True
    ) -> List[TrainingMetrics]:
        """Main training loop."""

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
```

### Reward API

#### BaseReward

```python
class BaseReward(ABC):
    """Abstract base class for reward functions."""

    def compute(self, **kwargs) -> float:
        """Compute the reward."""

    def compute_with_info(self, **kwargs) -> RewardInfo:
        """Compute reward with detailed information."""

    def normalize(
        self,
        reward: float,
        min_reward: float = -1.0,
        max_reward: float = 1.0
    ) -> float:
        """Normalize reward to specified range."""

    def clip(
        self,
        reward: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> float:
        """Clip reward to specified bounds."""
```

#### RewardInfo

```python
@dataclass
class RewardInfo:
    reward: float                                  # Computed reward
    components: Dict[str, float]                   # Breakdown
    metadata: Dict[str, Any]                       # Additional info
```

### Evaluation API

#### Evaluator

```python
class Evaluator:
    """Main evaluator for RL-LLM Agent."""

    def evaluate(
        self,
        num_episodes: Optional[int] = None,
        deterministic: Optional[bool] = None,
        verbose: bool = True
    ) -> EvaluationResult:
        """Run evaluation."""

    def compare_with_baseline(
        self,
        baseline_results: List[float],
        metric: str = "reward"
    ) -> Dict[str, float]:
        """Compare with baseline results."""

    def get_training_curve(
        self,
        window_size: int = 10
    ) -> Dict[str, List[float]]:
        """Get training/learning curve."""

    def export_results(self, filename: Optional[str] = None) -> str:
        """Export all results to file."""
```

#### MetricsCollector

```python
class MetricsCollector:
    """Collects and computes evaluation metrics."""

    def start_episode(self, episode_id: Optional[int] = None) -> int:
        """Start a new episode."""

    def record_step(
        self,
        reward: float,
        observation: Optional[str] = None,
        action: Optional[str] = None,
        **kwargs
    ) -> None:
        """Record a step within an episode."""

    def end_episode(
        self,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EpisodeMetrics:
        """End the current episode."""

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Get aggregate metrics across all episodes."""

    def get_reward_curve(self) -> List[float]:
        """Get reward curve."""

    def clear(self) -> None:
        """Clear all metrics."""
```

### Tools API

#### ToolRegistry

```python
class ToolRegistry:
    """Registry for tool discovery and management."""

    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: str = "general",
        examples: Optional[List[str]] = None,
    ) -> Callable:
        """Decorator to register a tool."""

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""

    def list_tools(self) -> List[str]:
        """List all registered tool names."""

    def parse_tool_call(self, text: str) -> Optional[Tuple[str, Dict]]:
        """Parse tool call from text."""
```

#### ToolManager

```python
class ToolManager:
    """Manager for tool execution."""

    def execute(
        self,
        tool_name: str,
        **kwargs
    ) -> ToolResult:
        """Execute a tool."""

    def parse_and_execute(self, text: str) -> Optional[ToolResult]:
        """Parse tool call from text and execute."""

    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get execution history."""
```

#### ToolResult

```python
@dataclass
class ToolResult:
    success: bool                                  # Success status
    output: Any                                    # Tool output
    error: Optional[str]                           # Error message
    metadata: Dict[str, Any]                       # Additional info
```

## Quick Reference

### Creating an Environment

```python
from environment.code_env import CodeEnvironment

env = CodeEnvironment({
    "max_steps": 30,
    "sandbox": True,
})

obs = env.reset()
action = Action(action_type="generate", value="def foo(): pass")
result = env.step(action)
```

### Using the Policy

```python
from agent.policy import PolicyNetwork, PolicyConfig
from agent.llm_wrapper import LLMConfig

config = PolicyConfig(
    llm_config=LLMConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct")
)
policy = PolicyNetwork(config)

# Get action
action, log_prob = policy.get_action_with_log_prob(obs_tensor)

# Get value
value = policy.get_value(obs_tensor)
```

### Training with PPO

```python
from rl.trainer import RLTrainer, TrainingConfig
from rl.ppo_trainer import PPOConfig

training_config = TrainingConfig(
    algorithm="ppo",
    total_steps=10000,
)

ppo_config = PPOConfig(lr=3e-5, epochs=4)

trainer = RLTrainer(
    policy=policy,
    env=env,
    config=training_config,
    ppo_config=ppo_config,
)

trainer.train()
```

### Evaluating

```python
from evaluation.evaluator import Evaluator, EvaluatorConfig

eval_config = EvaluatorConfig(num_episodes=100)

evaluator = Evaluator(
    policy=policy,
    env=env,
    config=eval_config,
)

result = evaluator.evaluate()
print(f"Success Rate: {result.success_rate:.2%}")
```
