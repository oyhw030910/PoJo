# Agent 示例集合

本目录包含 RL-LLM Agent Framework 的各种 Agent 实现示例。

## 目录结构

```
examples/
├── agent_example.py          # 通用 Agent 示例
├── code_agent.py             # 代码生成 Agent
├── math_agent.py             # 数学推理 Agent
├── run_agent_demo.py         # 交互式演示脚本
└── AGENT_GUIDE.md            # 详细使用指南
```

## 快速开始

### 1. 运行演示脚本

```bash
# 运行所有演示
python examples/run_agent_demo.py all

# 运行特定 Agent 演示
python examples/run_agent_demo.py code      # 代码 Agent
python examples/run_agent_demo.py math      # 数学 Agent
python examples/run_agent_demo.py chat      # 对话 Agent
python examples/run_agent_demo.py task      # 任务规划 Agent
python examples/run_agent_demo.py memory    # 记忆系统 Agent
```

### 2. 使用代码生成 Agent

```python
from examples.code_agent import CodeAgent

# 创建 Agent
agent = CodeAgent()

# 生成代码
result = agent.generate_code(
    description="编写一个函数计算斐波那契数列",
    test_cases=[
        {"args": [0], "expected": 0},
        {"args": [1], "expected": 1},
        {"args": [10], "expected": 55},
    ]
)

print(f"生成的代码：\n{result['code']}")
print(f"测试通过：{result['test_results']['passed']}/{result['test_results']['total']}")
```

### 3. 使用数学推理 Agent

```python
from examples.math_agent import MathAgent

# 创建 Agent
agent = MathAgent()

# 解决数学问题
problems = [
    ("What is 25 + 17?", "arithmetic"),
    ("What is 15% of 80?", "percentage"),
    ("Area of rectangle with length 8 and width 5?", "geometry"),
]

for problem, topic in problems:
    result = agent.solve(problem, topic=topic)
    print(f"{problem} = {result['answer']}")
```

### 4. 使用通用 Agent

```python
from examples.agent_example import LLMAgent, AgentConfig

# 创建 Agent
config = AgentConfig(
    use_memory=True,
    use_planner=True,
)
agent = LLMAgent(config)

# 运行任务
result = agent.run(task="解释什么是机器学习")
print(f"回复：{result['response']}")
```

### 5. 使用记忆系统

```python
from agent.memory import MemoryManager

# 创建记忆管理器
memory = MemoryManager()

# 添加经验
memory.add_experience(
    observation="用户询问天气",
    action="提供天气信息",
    reward=0.8,
)

# 获取上下文
context = memory.get_context(query="天气", num_recent=5)
print(context)
```

### 6. 使用规划器

```python
from agent.planner import Planner, PlannerConfig
from agent.llm_wrapper import LLMWrapper

# 创建组件
llm = LLMWrapper()
planner = Planner(llm, config=PlannerConfig(method="react"))

# 生成计划
result = planner.plan("准备一顿健康的早餐")

# 执行计划
for thought in result.plan.thoughts:
    print(f"步骤：{thought.content}")
    if thought.action:
        print(f"  动作：{thought.action}")
```

## Agent 类型详解

### CodeAgent（代码生成 Agent）

**用途**: 代码生成、调试、优化

**功能**:
- 从描述生成代码
- 自动语法检查
- 测试用例验证
- 迭代优化
- 代码解释
- 调试修复

**配置选项**:
```python
CodeAgentConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    max_iterations=5,      # 最大迭代次数
    auto_test=True,        # 自动测试
    use_lora=False,        # 是否使用 LoRA
)
```

### MathAgent（数学推理 Agent）

**用途**: 数学问题求解

**支持类型**:
- 算术运算
- 百分比计算
- 代数方程
- 几何问题

**功能**:
- 分步推理
- 计算器集成
- 答案验证
- 多策略求解

**配置选项**:
```python
MathAgentConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    step_by_step=True,     # 显示步骤
    use_calculator=True,   # 使用计算器
    max_steps=10,          # 最大步骤数
)
```

### LLMAgent（通用 Agent）

**用途**: 通用任务处理

**功能**:
- 对话交互
- 任务规划
- 工具使用
- 记忆管理

**配置选项**:
```python
AgentConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    use_memory=True,       # 启用记忆
    use_planner=True,      # 启用规划
    max_steps=50,          # 最大步数
    lora_enabled=False,    # 是否使用 LoRA
)
```

## 自定义 Agent

### 创建对话 Agent

```python
from agent.llm_wrapper import LLMWrapper, LLMConfig

class ChatAgent:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.llm = LLMWrapper(LLMConfig(model_name=model_name))
        self.history = []

    def chat(self, message):
        self.history.append({"role": "user", "content": message})

        prompt = self._build_prompt()
        response = self.llm.generate(prompt, max_new_tokens=256)

        self.history.append({"role": "assistant", "content": response})
        return response

    def _build_prompt(self):
        prompt = "You are a helpful assistant.\n\n"
        for msg in self.history[-10:]:
            prompt += f"{msg['role']}: {msg['content']}\n"
        prompt += "assistant: "
        return prompt

    def reset(self):
        self.history = []

# 使用
agent = ChatAgent()
while True:
    user = input("You: ")
    if user == "quit":
        break
    print(f"Agent: {agent.chat(user)}")
```

### 创建多工具 Agent

```python
from agent.llm_wrapper import LLMWrapper
from agent.memory import MemoryManager
from agent.tool_manager import ToolManager

class MultiToolAgent:
    def __init__(self):
        self.llm = LLMWrapper()
        self.memory = MemoryManager()
        self.tools = ToolManager()

    def process(self, request):
        # 决定使用哪个工具
        tool_decision = self._decide_tool(request)

        if tool_decision:
            tool_name, params = tool_decision
            result = self.tools.execute(tool_name, **params)
            return f"Result: {result.output}"
        else:
            return self.llm.generate(request)

    def _decide_tool(self, request):
        # 简化的工具选择逻辑
        if "calculate" in request.lower():
            return "calculate", {"expression": request.split()[-1]}
        if "code" in request.lower():
            return "execute_code", {"code": request}
        return None

# 使用
agent = MultiToolAgent()
response = agent.process("Calculate 15 * 27")
print(response)
```

## 高级技巧

### 批量处理任务

```python
def batch_process(agent, tasks, batch_size=5):
    results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        for task in batch:
            result = agent.solve(task)
            results.append(result)
        print(f"Processed batch {i // batch_size + 1}")
    return results
```

### 评估 Agent 性能

```python
def evaluate_agent(agent, test_tasks):
    successes = 0
    total_reward = 0

    for task in test_tasks:
        result = agent.solve(task)
        if result.get("success", False):
            successes += 1
        total_reward += result.get("reward", 0)

    return {
        "success_rate": successes / len(test_tasks),
        "avg_reward": total_reward / len(test_tasks),
    }
```

### Agent 记忆优化

```python
from agent.memory import MemoryConfig, MemoryManager

# 配置长短期记忆
config = MemoryConfig(
    short_term_size=20,
    long_term_enabled=True,
    long_term_top_k=5,
)
memory = MemoryManager(config)

# 添加高重要性经验
memory.add_experience(
    observation="复杂任务完成",
    action="使用分步策略",
    reward=1.0,
    importance=0.9,
)
```

## 规划策略选择

| 策略 | 用途 | 配置 |
|------|------|------|
| ReAct | 简单任务 | `method="react"` |
| Tree of Thoughts | 复杂推理 | `method="tot"` |
| Reflection | 需要改进 | `method="reflection"` |
| Decomposition | 多步骤任务 | `method="decomposition"` |

## 模型选择建议

| 模型大小 | 用途 | 资源需求 |
|----------|------|----------|
| 0.5B-1.5B | 快速实验 | 低 |
| 3B-7B | 平衡性能 | 中 |
| 8B+ | 高质量输出 | 高 |

## 常见问题

### Q: 如何提高 Agent 性能？
1. 启用 LoRA 微调
2. 增加记忆容量
3. 使用适当的规划策略
4. 添加领域特定工具

### Q: Agent 训练需要多少数据？
- 简单任务：100-1000 样本
- 复杂任务：1000-10000 样本
- 专业领域：可能需要更多

### Q: 如何处理长上下文？
- 使用记忆系统管理上下文
- 启用长期记忆检索
- 调整短期记忆大小

## 参考资源

- `AGENT_GUIDE.md` - 详细使用指南
- `../docs/api.md` - API 文档
- `../docs/tutorial.md` - 教程
- `../README.md` - 项目文档
