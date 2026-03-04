# RL-LLM Agent Tutorial

This tutorial walks you through using the RL-LLM Agent Framework.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Understanding the Components](#understanding-the-components)
4. [Training Your First Agent](#training-your-first-agent)
5. [Custom Environments](#custom-environments)
6. [Custom Rewards](#custom-rewards)
7. [Advanced Features](#advanced-features)

## Installation

### Basic Installation

```bash
# Clone or download the project
cd rl_llm_agent

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import rl_llm_agent; print('Installation successful!')"
```

### Dependencies

Key dependencies:
- PyTorch >= 2.0.0
- transformers >= 4.35.0
- LangChain >= 0.1.0
- gymnasium >= 0.29.0

## Quick Start

### Running a Demo

```bash
# Run interactive code environment demo
python main.py demo --env code
```

This launches an interactive session where you can:
- Generate code
- Execute code
- Submit solutions
- See rewards and feedback

### Simple Training

```bash
# Train on code tasks
python main.py train --env code --algorithm ppo --total-steps 1000
```

## Understanding the Components

### Environment

The environment defines how the agent interacts with the task:

```python
from environment.code_env import CodeEnvironment, CodeTask

# Create environment
env = CodeEnvironment({"max_steps": 30})

# Define a task
task = CodeTask(
    id="add_task",
    description="Write a function to add two numbers",
    starter_code="def add(a, b):\n    ",
    signature="def add(a, b)",
    test_cases=[
        {"args": [1, 2], "expected": 3},
        {"args": [5, 7], "expected": 12},
    ]
)

# Reset with task
obs = env.reset(task=task)
print(obs.text)
```

### Agent

The agent consists of several components:

```python
from agent.policy import PolicyNetwork, PolicyConfig
from agent.llm_wrapper import LLMConfig, LLMWrapper
from agent.memory import MemoryManager
from agent.planner import Planner

# Create LLM wrapper
llm = LLMWrapper(LLMConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    lora_enabled=True,
))

# Create policy
policy = PolicyNetwork(PolicyConfig(llm_config=LLMConfig()))

# Create memory manager
memory = MemoryManager()

# Create planner
planner = Planner(llm_wrapper=llm)
```

### RL Trainer

```python
from rl.trainer import RLTrainer, TrainingConfig
from rl.ppo_trainer import PPOConfig

# Configure training
training_config = TrainingConfig(
    algorithm="ppo",
    total_steps=10000,
    eval_interval=500,
)

# Configure PPO
ppo_config = PPOConfig(
    lr=3e-5,
    epochs=4,
    batch_size=64,
)

# Create trainer
trainer = RLTrainer(
    policy=policy,
    env=env,
    config=training_config,
    ppo_config=ppo_config,
)
```

## Training Your First Agent

### Step 1: Prepare Configuration

Create a config file `my_config.yaml`:

```yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  lora:
    enabled: true
    r: 32
    alpha: 64

ppo:
  lr: 3e-5
  epochs: 3
  batch_size: 32

training:
  total_steps: 5000
  eval_interval: 200
```

### Step 2: Start Training

```bash
python main.py train --env code --config my_config.yaml
```

### Step 3: Monitor Training

Training logs are saved to `./outputs/logs/`. View with TensorBoard:

```bash
tensorboard --logdir ./outputs/logs
```

### Step 4: Evaluate

```bash
python main.py eval --checkpoint ./outputs/checkpoints/final_model.pt --env code
```

## Custom Environments

### Creating a Custom Environment

```python
from environment.base_env import (
    BaseEnvironment, Observation, Action, StepResult,
    ActionSpace, ObservationSpace
)

class SimpleEnv(BaseEnvironment):
    """Simple number guessing game."""

    def __init__(self, config=None):
        super().__init__(config)
        self._target = None
        self._guesses = []

    def reset(self, seed=None, **kwargs):
        if seed:
            self._rng = np.random.default_rng(seed)
        self._target = self._rng.integers(1, 100)
        self._guesses = []
        self._current_step = 0
        self._done = False
        return Observation(text=f"Guess a number between 1 and 100")

    def step(self, action: Action) -> StepResult:
        self._current_step += 1
        guess = int(action.value)
        self._guesses.append(guess)

        if guess == self._target:
            reward = 1.0
            done = True
            info = {"success": True}
        elif guess < self._target:
            reward = -0.1
            done = False
            info = {"hint": "higher"}
        else:
            reward = -0.1
            done = False
            info = {"hint": "lower"}

        # Truncate after max steps
        truncated = self._current_step >= self._max_steps
        if truncated:
            done = True
            reward = -1.0

        return StepResult(
            observation=Observation(text=f"Guess {guess}: {info.get('hint', 'correct')}"),
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        )

    def get_action_space(self) -> ActionSpace:
        return ActionSpace(
            action_types=["guess"],
            value_space="discrete",
            constraints={"min": 1, "max": 100}
        )

    def get_observation_space(self) -> ObservationSpace:
        return ObservationSpace(text_max_length=256)

    def compute_reward(self, **kwargs) -> float:
        return 0.0

    def get_info(self) -> dict:
        return {"target": self._target, "guesses": self._guesses}

    def is_valid_action(self, action: Action) -> bool:
        if action.action_type != "guess":
            return False
        try:
            guess = int(action.value)
            return 1 <= guess <= 100
        except:
            return False

    def get_task_description(self) -> str:
        return "Guess the number between 1 and 100"
```

### Using Custom Environment

```python
env = SimpleEnv({"max_steps": 10})
obs = env.reset()

while True:
    action = Action(action_type="guess", value=50)
    result = env.step(action)
    print(f"Reward: {result.reward}, Info: {result.info}")
    if result.done:
        break
```

## Custom Rewards

### Creating a Custom Reward Function

```python
from reward.base_reward import BaseReward, RewardInfo

class EfficiencyReward(BaseReward):
    """Reward based on efficiency."""

    def compute(self, **kwargs) -> float:
        steps = kwargs.get("steps", 1)
        success = kwargs.get("success", False)

        if not success:
            return -1.0

        # More efficient = higher reward
        optimal = kwargs.get("optimal_steps", 1)
        efficiency = min(1.0, optimal / steps)

        return efficiency

    def compute_with_info(self, **kwargs) -> RewardInfo:
        reward = self.compute(**kwargs)
        return RewardInfo(
            reward=reward,
            components={
                "efficiency": reward,
            },
            metadata={
                "steps": kwargs.get("steps", 0),
                "success": kwargs.get("success", False),
            }
        )
```

### Using Composite Rewards

```python
from reward.composite_reward import CompositeReward
from reward.task_reward import TaskReward

# Create composite reward
composite = CompositeReward()

# Add task completion reward
composite.add_component("task", TaskReward(), weight=1.0)

# Add efficiency reward
composite.add_component("efficiency", EfficiencyReward(), weight=0.1)

# Compute reward
reward = composite.compute(
    task_kwargs={"success": True},
    efficiency_kwargs={"steps": 5, "success": True, "optimal_steps": 3}
)
```

## Advanced Features

### Using Memory

```python
from agent.memory import MemoryManager

memory = MemoryManager()

# Add experiences
memory.add_experience(
    observation="Code generated",
    action="def add(): pass",
    reward=0.5,
    store_long_term=True,
)

# Get context for decision making
context = memory.get_context(
    query="code generation",
    include_short_term=True,
    include_long_term=True,
    num_recent=5,
)
print(context)
```

### Using Planning

```python
from agent.planner import Planner, PlannerConfig

# Create planner with ReAct
planner = Planner(
    llm_wrapper=llm,
    config=PlannerConfig(method="react", max_iterations=10)
)

# Generate plan
result = planner.plan(
    task="Solve this math problem: 2x + 3 = 11",
    env=env,
)

# Execute plan
for thought in result.plan.thoughts:
    print(f"Thought: {thought.content}")
    if thought.action:
        print(f"  Action: {thought.action}")
```

### Using Tools

```python
from agent.tool_manager import ToolManager, ToolRegistry

# Create tool registry
registry = ToolRegistry()

# Register a tool
@registry.register(name="multiply", description="Multiply two numbers")
def multiply(a: float, b: float) -> float:
    return a * b

# Create manager and execute
manager = ToolManager(registry)
result = manager.execute("multiply", a=3, b=4)
print(f"Result: {result.output}")  # 12
```

### LangChain Integration

```python
from tools.langchain_tools import LangChainTools

# Create LangChain tools
lc_tools = LangChainTools()

# Load tools (requires langchain installation)
loaded = lc_tools.load_tools(["python_repl_ast", "llm-math"])

# Execute
result = lc_tools.execute("python_repl", code="print(2 + 2)")
print(result)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller model
   - Enable gradient accumulation

2. **Training Unstable**
   - Reduce learning rate
   - Increase clip epsilon
   - Add more entropy regularization

3. **Poor Performance**
   - Increase training steps
   - Tune reward shaping
   - Try different algorithm (PPO vs GRPO)

### Getting Help

- Check the [Architecture Documentation](architecture.md)
- Review [API Documentation](api.md)
- Look at test files for examples
