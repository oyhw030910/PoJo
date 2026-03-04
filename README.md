# RL-LLM Agent Framework 使用手册

## 目录

1. [简介](#简介)
2. [安装指南](#安装指南)
3. [快速入门](#快速入门)
4. [Agent 示例](#agent-示例)
5. [命令行工具](#命令行工具)
6. [配置详解](#配置详解)
7. [环境使用](#环境使用)
8. [训练指南](#训练指南)
9. [评估指南](#评估指南)
10. [高级功能](#高级功能)
11. [故障排除](#故障排除)

---

## 简介

RL-LLM Agent Framework 是一个用于训练大语言模型（LLM）智能体的强化学习框架。它支持多种 RL 算法（PPO、GRPO）和多种任务环境（代码生成、数学推理、GUI 导航）。

### 核心特性

| 特性 | 描述 |
|------|------|
| 多算法支持 | PPO、GRPO 强化学习算法 |
| 多环境支持 | 代码、数学、GUI 三种环境 |
| LLM 集成 | 支持 Qwen、Llama 等开源模型 |
| LoRA 微调 | 参数高效微调 |
| 记忆系统 | 短期和长期记忆管理 |
| 规划策略 | ReAct、思维树、反思 |
| 工具系统 | 集成 LangChain 工具 |

---

## 安装指南

### 系统要求

- Python 3.10+ (Python 3.9 也可用于基本功能)
- PyTorch 2.0+
- CUDA 11.7+ (GPU 训练)
- 至少 16GB RAM
- 推荐 24GB+ GPU 显存

### 快速安装

```bash
# 1. 进入项目目录
cd rl_llm_agent

# 2. 安装核心依赖（仅使用 Agent 功能）
pip install torch transformers peft

# 3. 验证安装
python -c "from examples.code_agent import CodeAgent; print('安装成功！')"
```

### 完整安装（包含 RL 训练功能）

```bash
# 1. 克隆或进入项目目录
cd rl_llm_agent

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 3. 安装所有依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import rl_llm_agent; print('安装成功！')"
```

### 依赖说明

| 依赖 | 用途 |
|------|------|
| torch | 深度学习框架 |
| transformers | HuggingFace 模型库 |
| peft | LoRA 微调 |
| langchain | 工具集成 |
| gymnasium | 环境接口 |
| tensorboard | 训练监控 |

---

## 快速入门

### 1. 运行交互式演示

```bash
python main.py demo --env code
```

演示中可用的命令：
- `generate: <代码>` - 生成代码
- `execute` - 执行代码
- `submit` - 提交答案
- `reset` - 重新开始
- `quit` - 退出

### 2. 开始训练

```bash
# 代码环境训练
python main.py train --env code --algorithm ppo

# 数学环境训练
python main.py train --env math --algorithm grpo
```

### 3. 评估模型

```bash
python main.py eval --checkpoint ./outputs/checkpoints/final_model.pt --env code
```

---

## Agent 示例

项目提供了多种预建的 Agent 示例，位于 `examples/` 目录：

### 运行 Agent 演示

```bash
# 运行所有演示
python examples/run_agent_demo.py all

# 运行特定 Agent 演示
python examples/run_agent_demo.py code      # 代码生成 Agent
python examples/run_agent_demo.py math      # 数学推理 Agent
python examples/run_agent_demo.py chat      # 对话 Agent
python examples/run_agent_demo.py task      # 任务规划 Agent
python examples/run_agent_demo.py memory    # 记忆系统 Agent
```

### 代码生成 Agent

```python
from examples.code_agent import CodeAgent

agent = CodeAgent()
result = agent.generate_code(
    description="编写一个函数计算斐波那契数列",
    test_cases=[
        {"args": [0], "expected": 0},
        {"args": [10], "expected": 55},
    ]
)
print(result["code"])
```

### 数学推理 Agent

```python
from examples.math_agent import MathAgent

agent = MathAgent()
result = agent.solve("What is 25% of 80?")
print(f"答案：{result['answer']}")
```

### 通用 Agent

```python
from examples.agent_example import LLMAgent, AgentConfig

config = AgentConfig(use_memory=True, use_planner=True)
agent = LLMAgent(config)
agent.run(task="解释什么是机器学习")
```

更多示例请参考：
- `examples/README.md` - Agent 示例说明
- `examples/AGENT_GUIDE.md` - 详细 Agent 使用指南

---

## 命令行工具

### train 命令

训练模型的完整参数：

```bash
python main.py train [选项]

选项:
  --env           环境类型 (code/math/gui), 默认：code
  --algorithm     算法类型 (ppo/grpo), 默认：ppo
  --config        配置文件路径, 默认：config/train_config.yaml
  --output-dir    输出目录，默认：./outputs
  --model         模型名称或路径
  --total-steps   总训练步数
  --lr            学习率
  --seed          随机种子，默认：42
  --device        设备 (cuda/cpu), 默认：cuda
  --resume        恢复检查点路径
```

### eval 命令

评估模型的完整参数：

```bash
python main.py eval [选项]

选项:
  --checkpoint     检查点路径（必需）
  --env            环境类型，默认：code
  --config         配置文件路径
  --output-dir     输出目录
  --num-episodes   评估集数，默认：100
  --deterministic  使用确定性动作
  --device         设备
```

### demo 命令

运行演示的参数：

```bash
python main.py demo [选项]

选项:
  --env    环境类型
  --model  模型检查点路径
```

---

## 配置详解

### 配置文件结构

```
config/
├── default_config.yaml      # 默认配置
├── train_config.yaml        # 训练配置
├── eval_config.yaml         # 评估配置
└── environment/
    ├── code_env.yaml        # 代码环境配置
    ├── math_env.yaml        # 数学环境配置
    └── gui_env.yaml         # GUI 环境配置
```

### 关键配置项

#### 模型配置

```yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"  # HuggingFace 模型名
  dtype: "float16"                     # 数据类型
  device: "cuda"                       # 运行设备
  max_seq_length: 2048                 # 最大序列长度

  lora:                                # LoRA 配置
    enabled: true
    r: 64                              # LoRA 秩
    alpha: 128                         # LoRA alpha
    dropout: 0.1                       # Dropout 率
    target_modules:                    # 目标模块
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
```

#### PPO 配置

```yaml
ppo:
  lr: 3e-5                             # 学习率
  betas: [0.9, 0.999]                  # Adam betas
  epochs: 4                            # PPO 轮数
  batch_size: 64                       # 批次大小
  mini_batch_size: 32                  # 小批次大小
  gamma: 0.99                          # 折扣因子
  lam: 0.95                            # GAE lambda
  clip_epsilon: 0.2                    # PPO clip 范围
  clip_value_loss: true                # 是否 clip 价值损失
  value_coef: 0.5                      # 价值损失系数
  entropy_coef: 0.01                   # 熵系数
  max_grad_norm: 0.5                   # 最大梯度范数
```

#### 训练配置

```yaml
training:
  total_steps: 100000                  # 总训练步数
  rollout_steps: 2048                  # 每次收集步数
  eval_interval: 1000                  # 评估间隔
  save_interval: 5000                  # 保存间隔
  log_interval: 100                    # 日志间隔
  checkpoint_dir: "./checkpoints"      # 检查点目录
  log_dir: "./logs"                    # 日志目录

  early_stopping: true                 # 早停
  patience: 5                          # 早停耐心度
  min_delta: 0.001                     # 最小改进量
```

#### 环境配置

```yaml
environment:
  name: "code"                         # 环境名称
  max_steps: 30                        # 最大步数
  timeout: 300                         # 超时时间 (秒)

  code:                                # 代码环境特定配置
    sandbox: true                      # 沙箱执行
    allowed_imports:                   # 允许的导入
      - "math"
      - "numpy"
      - "re"
```

---

## 环境使用

### 代码环境

用于代码生成和执行任务。

```python
from environment.code_env import CodeEnvironment, CodeTask

# 创建环境
env = CodeEnvironment({"max_steps": 30})

# 创建任务
task = CodeTask(
    id="add_task",
    description="编写一个函数，将两个数相加",
    starter_code="def add(a, b):\n    ",
    signature="def add(a, b)",
    test_cases=[
        {"args": [1, 2], "expected": 3},
        {"args": [5, 7], "expected": 12},
    ]
)

# 开始
obs = env.reset(task=task)
print(obs.text)

# 执行动作
from environment.base_env import Action
action = Action(action_type="generate", value="def add(a, b):\n    return a + b")
result = env.step(action)
print(f"奖励：{result.reward}")
```

### 数学环境

用于数学推理任务。

```python
from environment.math_env import MathEnvironment, MathTask

env = MathEnvironment({"max_steps": 20, "allow_calculator": True})

task = MathTask(
    id="math_001",
    problem="2 + 2 = ?",
    solution="2 + 2 = 4",
    answer="4",
    topic="arithmetic",
)

obs = env.reset(task=task)

# 推理动作
action = Action(action_type="reason", value="我们需要计算 2+2")
result = env.step(action)

# 计算器动作
action = Action(action_type="calculate", value="2 + 2")
result = env.step(action)
```

### GUI 环境

用于 GUI 导航任务。

```python
from environment.gui_env import GUIEnvironment, GUITask, GUIElement

env = GUIEnvironment({"max_steps": 50})

elements = [
    GUIElement(id="btn1", element_type="button", text="提交"),
    GUIElement(id="input1", element_type="input", text=""),
]

task = GUITask(
    id="gui_001",
    instruction="点击提交按钮",
    start_url="http://example.com",
    goal_description="提交按钮已点击",
    goal_check=lambda state: True,
    elements=elements,
)

obs = env.reset(task=task)

# 点击动作
action = Action(action_type="click", value="btn1")
result = env.step(action)
```

---

## 训练指南

### 基础训练

```bash
# 使用默认配置训练
python main.py train --env code

# 指定算法
python main.py train --env math --algorithm grpo

# 自定义学习率
python main.py train --env code --lr 1e-5
```

### 进阶训练

```bash
# 使用自定义配置
python main.py train --config my_config.yaml

# 指定输出目录
python main.py train --output-dir ./my_outputs

# 恢复训练
python main.py train --resume ./outputs/checkpoints/checkpoint_5000.pt
```

### 训练监控

```bash
# 使用 TensorBoard 查看训练
tensorboard --logdir ./outputs/logs

# 在浏览器中打开 http://localhost:6006
```

### 训练输出

训练完成后，输出目录结构：

```
outputs/
├── checkpoints/           # 模型检查点
│   ├── checkpoint_1000.pt
│   ├── checkpoint_5000.pt
│   └── final_model.pt
├── logs/                  # 日志文件
│   ├── tensorboard/
│   └── *.log
└── results/               # 评估结果
```

---

## 评估指南

### 基础评估

```bash
python main.py eval --checkpoint ./outputs/checkpoints/final_model.pt
```

### 高级评估

```bash
# 指定环境
python main.py eval --checkpoint model.pt --env math

# 增加评估集数
python main.py eval --checkpoint model.pt --num-episodes 200

# 随机策略评估
python main.py eval --checkpoint model.pt --no-deterministic
```

### 评估结果

评估结果保存在 `evaluation_results/` 目录：

```json
{
  "timestamp": "20260304_120000",
  "num_episodes": 100,
  "success_rate": 0.75,
  "avg_reward": 0.623,
  "avg_episode_length": 12.5,
  "metrics": {...},
  "task_metrics": {...}
}
```

---

## 高级功能

### 自定义环境

```python
from environment.base_env import BaseEnvironment, Observation, Action, StepResult

class MyEnv(BaseEnvironment):
    def reset(self, seed=None, **kwargs) -> Observation:
        self._state = 0
        return Observation(text="初始状态")

    def step(self, action: Action) -> StepResult:
        self._state += 1
        done = self._state >= 10
        return StepResult(
            observation=Observation(text=f"状态：{self._state}"),
            reward=1.0 if done else 0.1,
            done=done,
            truncated=False,
            info={}
        )

    def get_action_space(self) -> ActionSpace:
        return ActionSpace(action_types=["move"], value_space="discrete")

    def get_observation_space(self) -> ObservationSpace:
        return ObservationSpace(text_max_length=100)

    def compute_reward(self, **kwargs) -> float:
        return 0.0

    def get_info(self) -> dict:
        return {"state": self._state}

    def is_valid_action(self, action: Action) -> bool:
        return action.action_type == "move"

    def get_task_description(self) -> str:
        return "前进到状态 10"
```

### 自定义奖励函数

```python
from reward.base_reward import BaseReward, RewardInfo

class MyReward(BaseReward):
    def compute(self, **kwargs) -> float:
        success = kwargs.get("success", False)
        steps = kwargs.get("steps", 1)

        if not success:
            return -1.0

        # 效率奖励
        efficiency = min(1.0, 10 / steps)
        return efficiency

    def compute_with_info(self, **kwargs) -> RewardInfo:
        reward = self.compute(**kwargs)
        return RewardInfo(
            reward=reward,
            components={"efficiency": reward},
            metadata={"success": kwargs.get("success")}
        )
```

### 使用记忆系统

```python
from agent.memory import MemoryManager

memory = MemoryManager()

# 添加经验
memory.add_experience(
    observation="代码已生成",
    action="def add(): pass",
    reward=0.5,
    store_long_term=True,
)

# 获取上下文
context = memory.get_context(
    query="代码生成",
    num_recent=5,
)
print(context)
```

### 使用规划器

```python
from agent.planner import Planner, PlannerConfig

planner = Planner(
    llm_wrapper=llm,
    config=PlannerConfig(
        method="react",        # react/tot/reflection/decomposition
        max_iterations=10,
    )
)

# 生成计划
result = planner.plan(
    task="解决这个数学问题",
    env=env,
)

# 执行计划
for thought in result.plan.thoughts:
    print(f"思考：{thought.content}")
    if thought.action:
        print(f"  动作：{thought.action}")
```

### 使用工具

```python
from agent.tool_manager import ToolManager, ToolRegistry

# 注册工具
registry = ToolRegistry()

@registry.register(name="multiply", description="两数相乘")
def multiply(a: float, b: float) -> float:
    return a * b

# 执行工具
manager = ToolManager(registry)
result = manager.execute("multiply", a=3, b=4)
print(f"结果：{result.output}")  # 12
```

---

## 故障排除

### 常见问题

#### 1. CUDA 内存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```yaml
# 减小批次大小
ppo:
  batch_size: 16
  mini_batch_size: 4

# 使用更小的模型
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"
```

#### 2. 训练不稳定

**症状**: 奖励波动大，KL 散度过高

**解决方案**:
```yaml
# 降低学习率
ppo:
  lr: 1e-5

# 增加熵系数
ppo:
  entropy_coef: 0.02

# 减小 clip 范围
ppo:
  clip_epsilon: 0.1
```

#### 3. 性能差

**症状**: 成功率低，收敛慢

**解决方案**:
- 增加训练步数
- 调整奖励塑形
- 使用课程学习
- 尝试不同算法（PPO vs GRPO）

#### 4. 导入错误

**症状**: `ModuleNotFoundError`

**解决方案**:
```bash
# 重新安装依赖
pip install -r requirements.txt --force-reinstall

# 检查 Python 版本
python --version  # 需要 3.10+
```

### 日志位置

| 日志类型 | 位置 |
|----------|------|
| 训练日志 | `./outputs/logs/*.log` |
| TensorBoard | `./outputs/logs/tensorboard/` |
| 评估结果 | `./evaluation_results/` |
| 检查点 | `./outputs/checkpoints/` |

### 获取帮助

1. 查看文档：`docs/` 目录
2. 运行测试：`pytest tests/ -v`
3. 查看示例：`experiments/` 目录

---

## 附录

### 支持的模型

| 模型 | 大小 | 推荐用途 |
|------|------|----------|
| Qwen2.5-0.5B | 0.5B | 快速实验 |
| Qwen2.5-1.5B | 1.5B | 默认推荐 |
| Qwen2.5-3B | 3B | 高质量输出 |
| Llama-3-8B | 8B | 高性能需求 |

### 推荐硬件配置

| 配置 | 最低 | 推荐 |
|------|------|------|
| GPU | RTX 3060 12GB | RTX 4090 24GB |
| RAM | 16GB | 32GB |
| 存储 | 50GB SSD | 100GB NVMe |

### 版本信息

- 框架版本：0.1.0
- Python 版本：3.10+
- PyTorch 版本：2.0+
