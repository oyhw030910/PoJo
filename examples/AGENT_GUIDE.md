# Agent 创建与使用指南

本指南详细说明如何基于 RL-LLM Agent Framework 创建和使用各种 Agent。

## 目录

1. [快速开始](#快速开始)
2. [基础 Agent 创建](#基础 agent 创建)
3. [代码生成 Agent](#代码生成 agent)
4. [数学推理 Agent](#数学推理 agent)
5. [自定义 Agent](#自定义 agent)
6. [Agent 与环境集成](#agent 与环境集成)
7. [高级技巧](#高级技巧)

---

## 快速开始

### 最简单的 Agent

```python
from agent.llm_wrapper import LLMWrapper, LLMConfig

# 1. 创建 LLM 包装器
llm = LLMWrapper(LLMConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct"
))

# 2. 生成文本
response = llm.generate("解释什么是机器学习")
print(response)
```

### 带记忆的 Agent

```python
from agent.llm_wrapper import LLMWrapper, LLMConfig
from agent.memory import MemoryManager

# 创建组件
llm = LLMWrapper(LLMConfig())
memory = MemoryManager()

# 添加经验
memory.add_experience(
    observation="用户询问天气",
    action="提供天气信息",
    reward=0.8,
)

# 获取上下文
context = memory.get_context()
print(context)
```

---

## 基础 Agent 创建

### 对话 Agent

```python
from agent.llm_wrapper import LLMWrapper, LLMConfig
from typing import List, Dict

class ChatAgent:
    """简单的对话 Agent"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.llm = LLMWrapper(LLMConfig(model_name=model_name))
        self.conversation_history: List[Dict] = []

    def chat(self, message: str) -> str:
        """进行对话"""
        # 添加到历史
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        # 构建提示
        prompt = self._build_prompt()

        # 生成回复
        response = self.llm.generate(prompt, max_new_tokens=256)

        # 保存回复
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def _build_prompt(self) -> str:
        """构建对话提示"""
        prompt = "You are a helpful assistant.\n\n"
        for msg in self.conversation_history[-10:]:  # 最近 10 条消息
            prompt += f"{msg['role']}: {msg['content']}\n"
        prompt += "assistant: "
        return prompt

    def reset(self):
        """重置对话历史"""
        self.conversation_history = []


# 使用示例
agent = ChatAgent()
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = agent.chat(user_input)
    print(f"Agent: {response}")
```

### 任务执行 Agent

```python
from agent.llm_wrapper import LLMWrapper, LLMConfig
from agent.planner import Planner, PlannerConfig

class TaskAgent:
    """任务执行 Agent"""

    def __init__(self):
        self.llm = LLMWrapper(LLMConfig())
        self.planner = Planner(
            self.llm,
            config=PlannerConfig(method="react", max_iterations=10)
        )

    def execute_task(self, task: str, steps: int = 5) -> list:
        """执行多步骤任务"""
        # 生成计划
        prompt = f"""Break down this task into {steps} steps:

Task: {task}

Provide a step-by-step plan:"""

        plan = self.llm.generate(prompt, max_new_tokens=300)

        # 执行步骤
        results = []
        for i, line in enumerate(plan.split('\n'), 1):
            if line.strip():
                results.append(f"Step {i}: {line}")

        return {
            "task": task,
            "plan": plan,
            "steps": results,
        }


# 使用示例
agent = TaskAgent()
result = agent.execute_task("准备一顿健康的早餐")
print(result["plan"])
```

---

## 代码生成 Agent

### 使用内置 CodeAgent

```python
from examples.code_agent import CodeAgent, CodeAgentConfig

# 创建 Agent
config = CodeAgentConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    max_iterations=3,
    auto_test=True,
)
agent = CodeAgent(config)

# 生成代码
result = agent.generate_code(
    description="编写一个函数，计算斐波那契数列的第 n 项",
    test_cases=[
        {"args": [1], "expected": 1},
        {"args": [5], "expected": 5},
        {"args": [10], "expected": 55},
    ],
    verbose=True,
)

print(f"成功：{result['success']}")
print(f"代码:\n{result['code']}")
```

### 自定义代码 Agent

```python
from agent.llm_wrapper import LLMWrapper, LLMConfig
from tools.code_tool import execute_code, check_code_syntax

class CustomCodeAgent:
    """自定义代码生成 Agent"""

    def __init__(self):
        self.llm = LLMWrapper(LLMConfig())
        self.code_history = []

    def generate(self, description: str) -> str:
        """生成代码"""
        prompt = f"""Generate Python code for:

{description}

Requirements:
- Include docstring
- Handle edge cases
- Only output code

Code:"""

        code = self.llm.generate(prompt, max_new_tokens=512)

        # 提取代码
        if "```python" in code:
            start = code.find("```python") + 9
            end = code.find("```", start)
            code = code[start:end].strip()

        self.code_history.append(code)
        return code

    def test(self, code: str, test_input: any) -> any:
        """测试代码"""
        result = execute_code(code)
        return result

    def explain(self, code: str) -> str:
        """解释代码"""
        prompt = f"""Explain this code:

```python
{code}
```

Explain what it does step by step:"""

        return self.llm.generate(prompt, max_new_tokens=256)


# 使用示例
agent = CustomCodeAgent()

# 生成
code = agent.generate("一个函数，判断字符串是否为回文")
print(f"Generated:\n{code}")

# 测试
result = agent.test(code, "\"racecar\"")
print(f"Test result: {result}")

# 解释
explanation = agent.explain(code)
print(f"Explanation: {explanation}")
```

---

## 数学推理 Agent

### 使用内置 MathAgent

```python
from examples.math_agent import MathAgent, MathAgentConfig

# 创建 Agent
config = MathAgentConfig(
    step_by_step=True,
    use_calculator=True,
)
agent = MathAgent(config)

# 解决问题
result = agent.solve("What is 25% of 120?", topic="percentage")

print(f"答案：{result['answer']}")
print(f"步骤:")
for step in result['steps']:
    print(f"  - {step}")
```

### 自定义数学 Agent

```python
from agent.llm_wrapper import LLMWrapper, LLMConfig
from tools.search_tool import calculate

class MathSolverAgent:
    """数学求解 Agent"""

    def __init__(self):
        self.llm = LLMWrapper(LLMConfig())

    def solve(self, problem: str) -> dict:
        """解决数学问题"""
        # 分析题目类型
        problem_type = self._classify_problem(problem)

        # 选择解决策略
        if problem_type == "calculation":
            return self._solve_calculation(problem)
        elif problem_type == "word_problem":
            return self._solve_word_problem(problem)
        else:
            return self._solve_general(problem)

    def _classify_problem(self, problem: str) -> str:
        """分类问题类型"""
        if any(op in problem for op in ["+", "-", "*", "/", "=", "%"]):
            return "calculation"
        elif len(problem.split()) > 15:
            return "word_problem"
        return "general"

    def _solve_calculation(self, problem: str) -> dict:
        """直接计算"""
        # 提取表达式
        expr = problem.replace("=", "")
        result = calculate(expr)
        return {
            "answer": result.get("result"),
            "method": "direct_calculation",
        }

    def _solve_word_problem(self, problem: str) -> dict:
        """文字题"""
        prompt = f"""Solve this word problem step by step:

{problem}

Show your work and end with "Answer: [value]"
"""
        solution = self.llm.generate(prompt, max_new_tokens=300)

        # 提取答案
        import re
        match = re.search(r"Answer:\s*(\d+(?:\.\d+)?)", solution)
        answer = match.group(1) if match else "Unknown"

        return {
            "answer": answer,
            "solution": solution,
            "method": "llm_reasoning",
        }

    def _solve_general(self, problem: str) -> dict:
        """通用解法"""
        return self._solve_word_problem(problem)


# 使用示例
agent = MathSolverAgent()

# 计算题
result1 = agent.solve("25 * 4 + 10")
print(f"Calculation: {result1['answer']}")

# 文字题
result2 = agent.solve("小明有 15 个苹果，他给了小红 7 个，还剩几个？")
print(f"Word problem: {result2['answer']}")
```

---

## 自定义 Agent

### 多模态 Agent（文本 + 工具）

```python
from agent.llm_wrapper import LLMWrapper, LLMConfig
from agent.memory import MemoryManager
from agent.tool_manager import ToolManager

class MultiToolAgent:
    """多工具 Agent"""

    def __init__(self):
        self.llm = LLMWrapper(LLMConfig())
        self.memory = MemoryManager()
        self.tools = ToolManager()

    def process(self, request: str) -> str:
        """处理请求"""
        # 决定使用哪个工具
        tool_decision = self._decide_tool(request)

        if tool_decision:
            tool_name, params = tool_decision
            result = self.tools.execute(tool_name, **params)

            # 格式化结果
            response = self._format_result(result)
        else:
            # 直接回复
            response = self.llm.generate(request, max_new_tokens=256)

        # 记录经验
        self.memory.add_experience(
            observation=request,
            action=str(tool_decision),
            reward=0.5,
        )

        return response

    def _decide_tool(self, request: str) -> tuple:
        """决定使用哪个工具"""
        prompt = f"""Analyze this request and determine which tool to use:

Request: {request}

Available tools:
- execute_code: For running Python code
- calculate: For mathematical calculations
- search: For information lookup

Respond with: TOOL_NAME: param1=value1, param2=value2
Or respond with: NONE if no tool is needed."""

        decision = self.llm.generate(prompt, max_new_tokens=100)

        if "NONE" in decision:
            return None

        if ":" in decision:
            tool_name = decision.split(":")[0].strip()
            params_str = decision.split(":")[1].strip()

            # 简单解析参数
            params = {}
            for param in params_str.split(","):
                if "=" in param:
                    key, value = param.split("=")
                    params[key.strip()] = value.strip()

            return tool_name, params

        return None

    def _format_result(self, result) -> str:
        """格式化结果"""
        if result.success:
            return f"Result: {result.output}"
        return f"Error: {result.error}"


# 使用示例
agent = MultiToolAgent()

response = agent.process("Calculate 15 * 27")
print(response)

response = agent.process("What is the capital of France?")
print(response)
```

### 带强化学习的 Agent

```python
from agent.policy import PolicyNetwork, PolicyConfig
from agent.llm_wrapper import LLMConfig
from rl.trainer import RLTrainer, TrainingConfig
from environment.base_env import Action

class RLAgent:
    """带 RL 训练的 Agent"""

    def __init__(self):
        # 创建策略网络
        policy_config = PolicyConfig(
            llm_config=LLMConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct")
        )
        self.policy = PolicyNetwork(policy_config)

        # 训练配置
        training_config = TrainingConfig(
            algorithm="ppo",
            total_steps=10000,
        )

        self.trainer = None  # 在训练时创建
        self.training_history = []

    def train(self, env, steps: int = 10000):
        """在环境中训练"""
        training_config = TrainingConfig(
            algorithm="ppo",
            total_steps=steps,
        )

        self.trainer = RLTrainer(
            policy=self.policy,
            env=env,
            config=training_config,
        )

        history = self.trainer.train()
        self.training_history = history
        return history

    def act(self, observation, deterministic: bool = True) -> Action:
        """执行动作"""
        action = self.policy.get_action(
            observation,
            deterministic=deterministic,
        )
        return action

    def save(self, path: str):
        """保存模型"""
        self.policy.save_pretrained(path)

    def load(self, path: str):
        """加载模型"""
        self.policy = PolicyNetwork.from_pretrained(path)
```

---

## Agent 与环境集成

### 代码环境 Agent

```python
from environment.code_env import CodeEnvironment, CodeTask
from agent.llm_wrapper import LLMWrapper, LLMConfig

class CodeEnvAgent:
    """代码环境专用 Agent"""

    def __init__(self):
        self.llm = LLMWrapper(LLMConfig())
        self.env = CodeEnvironment({"max_steps": 10})

    def solve_task(self, task: CodeTask) -> dict:
        """解决代码任务"""
        obs = self.env.reset(task=task)
        total_reward = 0

        for step in range(10):
            # 生成代码
            prompt = f"""{task.description}

Current observation:
{obs.text}

Generate the code:"""

            code = self.llm.generate(prompt, max_new_tokens=300)

            # 提交代码
            action = Action(action_type="generate", value=code)
            result = self.env.step(action)

            obs = result.observation
            total_reward += result.reward

            if result.done:
                break

        return {
            "task_id": task.id,
            "total_reward": total_reward,
            "steps": step + 1,
            "done": result.done,
        }


# 使用示例
agent = CodeEnvAgent()

task = CodeTask(
    id="test",
    description="编写函数判断一个数是否为偶数",
    starter_code="def is_even(n):\n    ",
    signature="def is_even(n)",
    test_cases=[
        {"args": [2], "expected": True},
        {"args": [3], "expected": False},
    ]
)

result = agent.solve_task(task)
print(f"奖励：{result['total_reward']}")
```

### 数学环境 Agent

```python
from environment.math_env import MathEnvironment, MathTask
from agent.llm_wrapper import LLMWrapper, LLMConfig

class MathEnvAgent:
    """数学环境专用 Agent"""

    def __init__(self):
        self.llm = LLMWrapper(LLMConfig())
        self.env = MathEnvironment({"max_steps": 5})

    def solve(self, problem: str) -> dict:
        """解决数学问题"""
        task = MathTask(
            id="problem",
            problem=problem,
            solution="",
            answer="",
        )

        obs = self.env.reset(task=task)
        total_reward = 0
        reasoning = ""

        for step in range(5):
            # 推理
            prompt = f"""{obs.text}

Think step by step:"""

            thought = self.llm.generate(prompt, max_new_tokens=200)
            reasoning += thought + "\n"

            # 提交答案
            action = Action(action_type="submit", value=thought.split()[-1])
            result = self.env.step(action)

            obs = result.observation
            total_reward += result.reward

            if result.done:
                break

        return {
            "answer": action.value,
            "reasoning": reasoning,
            "reward": total_reward,
            "correct": result.info.get("correct", False),
        }
```

---

## 高级技巧

### 1. Agent 记忆优化

```python
from agent.memory import MemoryManager, MemoryConfig

# 配置长短期记忆
config = MemoryConfig(
    short_term_size=20,      # 短期记忆容量
    long_term_enabled=True,  # 启用长期记忆
    long_term_top_k=5,       # 检索数量
)

memory = MemoryManager(config)

# 添加重要经验
memory.add_experience(
    observation="复杂任务完成",
    action="使用分步策略",
    reward=1.0,
    importance=0.9,  # 高重要性
)

# 检索相关记忆
context = memory.get_context(
    query="复杂任务",
    include_long_term=True,
    num_recent=10,
)
```

### 2. Agent 规划策略选择

```python
from agent.planner import Planner, PlannerConfig

# ReAct - 适合简单任务
react_planner = Planner(
    llm,
    config=PlannerConfig(method="react", max_iterations=5)
)

# Tree of Thoughts - 适合复杂推理
tot_planner = Planner(
    llm,
    config=PlannerConfig(method="tot", branching_factor=3, max_depth=2)
)

# Reflection - 适合需要改进的任务
reflection_planner = Planner(
    llm,
    config=PlannerConfig(method="reflection")
)
```

### 3. 批量处理

```python
def batch_process(agent, tasks: list, batch_size: int = 5):
    """批量处理任务"""
    results = []

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_results = []

        for task in batch:
            result = agent.solve(task)
            batch_results.append(result)

        results.extend(batch_results)
        print(f"Processed batch {i // batch_size + 1}")

    return results
```

### 4. Agent 评估

```python
def evaluate_agent(agent, test_tasks: list) -> dict:
    """评估 Agent 性能"""
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
        "total_tasks": len(test_tasks),
    }
```

---

## 常见问题

### Q: 如何选择模型大小？

- **0.5B-1.5B**: 快速实验，资源有限
- **3B-7B**: 平衡性能和速度
- **8B+**: 高质量输出，充足资源

### Q: 如何提高 Agent 性能？

1. 启用 LoRA 微调
2. 增加记忆容量
3. 使用适当的规划策略
4. 添加领域特定工具

### Q: Agent 训练需要多少数据？

- 简单任务：100-1000 个样本
- 复杂任务：1000-10000 个样本
- 专业领域：可能需要更多

---

## 参考资源

- `examples/agent_example.py` - 综合 Agent 示例
- `examples/code_agent.py` - 代码生成 Agent
- `examples/math_agent.py` - 数学推理 Agent
- `docs/api.md` - API 文档
- `docs/tutorial.md` - 详细教程
