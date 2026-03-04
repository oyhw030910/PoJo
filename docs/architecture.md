# Architecture Documentation

## System Overview

The RL-LLM Agent Framework consists of five main layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│              (Code/Math/GUI Tasks)                           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Agent Layer                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ Planner  │  │ Memory   │  │  Tools   │  │   Policy     │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                       RL Core Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │   PPO    │  │   GRPO   │  │  Reward  │  │   Trainer    │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Environment Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │   Code   │  │   Math   │  │   GUI    │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Tools Layer                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │  Code    │  │  Search  │  │LangChain │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### Agent Layer

#### LLM Wrapper (`agent/llm_wrapper.py`)
- Provides unified interface for various LLM backends
- Supports LoRA fine-tuning for parameter efficiency
- Handles tokenization and generation

#### Policy Network (`agent/policy.py`)
- Combines LLM backbone with actor-critic heads
- Computes action probabilities and value estimates
- Supports autoregressive action generation

#### Memory Manager (`agent/memory.py`)
- **Short-term memory**: Recent observations, actions, rewards
- **Long-term memory**: Vector store for important experiences
- **Working memory**: Current task context

#### Planner (`agent/planner.py`)
- **ReAct**: Alternating reasoning and action
- **Tree of Thoughts**: Exploring multiple reasoning paths
- **Reflection**: Learning from past experiences
- **Task Decomposition**: Breaking complex tasks into subtasks

#### Tool Manager (`agent/tool_manager.py`)
- Tool registration and discovery
- Tool execution and result parsing
- Integration with LangChain tools

### RL Core Layer

#### PPO Trainer (`rl/ppo_trainer.py`)
- Implements Proximal Policy Optimization
- GAE for advantage estimation
- Clipped surrogate objective
- Value function clipping

#### GRPO Trainer (`rl/grpo_trainer.py`)
- Implements Group Relative Policy Optimization
- No value function required
- Group-based advantage computation

#### Loss Functions (`rl/loss.py`)
- Policy loss (PPO-Clip)
- Value loss (MSE with clipping)
- Entropy bonus
- KL penalty
- GAE computation

#### Replay Buffer (`rl/replay_buffer.py`)
- Standard replay buffer
- Sequence buffer for on-policy learning
- Priority replay buffer

#### Trainer (`rl/trainer.py`)
- Main training loop orchestration
- Rollout collection
- Policy updates
- Checkpointing and logging

### Environment Layer

#### Base Environment (`environment/base_env.py`)
- Abstract base class defining the interface
- Observation, Action, StepResult dataclasses
- Standard Gym-like interface

#### Code Environment (`environment/code_env.py`)
- Code generation and execution
- Sandboxed code execution
- Test case verification
- Actions: generate, execute, test, submit

#### Math Environment (`environment/math_env.py`)
- Mathematical problem solving
- Calculator tool integration
- Step-by-step reasoning
- Actions: reason, calculate, verify, submit

#### GUI Environment (`environment/gui_env.py`)
- GUI navigation simulation
- Element interaction (click, type, etc.)
- Task completion detection
- Actions: click, type, navigate, scroll, select

### Reward Layer

#### Base Reward (`reward/base_reward.py`)
- Abstract base class for rewards
- Normalization and clipping utilities

#### Task Reward (`reward/task_reward.py`)
- Task completion rewards
- Partial completion support
- Multi-task reward aggregation

#### Shaping Reward (`reward/shaping_reward.py`)
- Step penalty for efficiency
- Format bonuses
- Intermediate step rewards
- Potential-based shaping

#### Composite Reward (`reward/composite_reward.py`)
- Combines multiple reward functions
- Weighted aggregation
- Curriculum-based rewards

### Evaluation Layer

#### Evaluator (`evaluation/evaluator.py`)
- Multi-episode evaluation
- Task-specific metrics
- Result saving and comparison

#### Metrics (`evaluation/metrics.py`)
- Episode metrics collection
- Aggregate statistics
- Success rate, reward curves
- Sample efficiency computation

### Tools Layer

#### Code Tools (`tools/code_tool.py`)
- Code execution
- Syntax checking
- Code analysis
- Test running

#### Search Tools (`tools/search_tool.py`)
- Knowledge base search
- Web search simulation
- Fact checking
- Calculations

#### LangChain Tools (`tools/langchain_tools.py`)
- LangChain integration
- Tool wrappers
- Pre-built tool sets

## Data Flow

### Training Flow

```
1. Environment generates observation
2. Agent processes observation through:
   - Memory retrieval
   - Planning (optional)
   - Policy network
3. Agent produces action
4. Environment executes action, returns:
   - New observation
   - Reward
   - Done flag
5. Store transition in buffer
6. Periodically update policy using RL algorithm
7. Log metrics and save checkpoints
```

### Inference Flow

```
1. Receive task description
2. Initialize environment
3. Loop until done:
   a. Get observation
   b. Retrieve relevant memories
   c. Plan next action (if using planner)
   d. Get action from policy
   e. Execute action
4. Return result
```

## Extension Points

### Adding Custom Environments

1. Inherit from `BaseEnvironment`
2. Implement required methods: `reset`, `step`, `get_observation`, `get_action_space`
3. Define custom observation and action spaces
4. Add reward computation logic

### Adding Custom Rewards

1. Inherit from `BaseReward`
2. Implement `compute()` method
3. Optionally implement `compute_with_info()` for detailed breakdown
4. Register with `CompositeReward`

### Adding Custom Tools

1. Define tool function with type hints
2. Use `@register_tool` decorator
3. Specify name, description, category, examples
4. Tool automatically available through `ToolManager`

### Adding Custom RL Algorithms

1. Create trainer class in `rl/` directory
2. Implement `update()` method for policy updates
3. Define configuration dataclass
4. Integrate with main `RLTrainer`

## Configuration System

Configuration follows a hierarchical structure:

```
default_config.yaml (base defaults)
       ↓
train_config.yaml (training overrides)
       ↓
Command line args (runtime overrides)
```

Key configuration sections:
- `model`: LLM settings
- `policy`: Policy network settings
- `ppo`/`grpo`: Algorithm hyperparameters
- `environment`: Environment settings
- `reward`: Reward configuration
- `training`: Training loop settings
- `evaluation`: Evaluation settings
