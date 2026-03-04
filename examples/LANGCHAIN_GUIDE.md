# LangChain 与 LangGraph Agent 开发指南

本指南展示如何使用 LangChain 和 LangGraph 框架**极大地简化**Agent 开发工作。

## 目录

1. [为什么使用 LangChain/LangGraph](#为什么使用-langchainlanggraph)
2. [快速开始](#快速开始)
3. [代码对比](#代码对比)
4. [LangChain 示例](#langchain-示例)
5. [LangGraph 示例](#langgraph-示例)
6. [最佳实践](#最佳实践)

---

## 为什么使用 LangChain/LangGraph

### 传统方式 (从零构建)

```python
# 需要 100+ 行代码
class CodeAgent:
    def __init__(self):
        self.llm = LLMWrapper(...)
        self.memory = MemoryManager()
        self.tools = ToolManager()
        self.planner = Planner(...)
        # ... 大量初始化代码

    def _build_prompt(self, task):
        # 手动构建提示
        pass

    def _call_tool(self, name, args):
        # 手动调用工具
        pass

    def _handle_errors(self, error):
        # 手动错误处理
        pass

    def run(self, task):
        # 手动管理工作流
        pass
```

### LangChain 方式

```python
# 只需 10 行代码
from langchain.agents import initialize_agent, load_tools

tools = load_tools(["python_repl", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react")
result = agent.run("Write code to calculate factorial of 5")
```

### 代码量对比

| 功能 | 从零构建 | LangChain | LangGraph |
|------|----------|-----------|-----------|
| 基础 Agent | 100+ 行 | 10 行 | 15 行 |
| 工具集成 | 50+ 行/工具 | 1 行/工具 | 1 行/工具 |
| 工作流管理 | 80+ 行 | 内置 | 内置 |
| 状态管理 | 手动 | 有限 | 完全内置 |
| 多 Agent | 200+ 行 | 复杂 | 简洁 |

---

## 快速开始

### 安装依赖

```bash
# 核心依赖
pip install langchain langchain-community

# LangGraph (用于复杂工作流)
pip install langgraph

# 可选：更多工具
pip install langchainhub wikipedia wolframalpha
```

### 验证安装

```bash
python examples/langchain_agent.py basic
python examples/langgraph_agent.py diagram
```

---

## 代码对比

### 示例：创建带工具的 Agent

#### 从零构建 (~80 行)

```python
from agent.llm_wrapper import LLMWrapper
from agent.tool_manager import ToolManager

class CalculatorAgent:
    def __init__(self):
        self.llm = LLMWrapper(...)
        self.tools = ToolManager()
        self._register_tools()

    def _register_tools(self):
        @self.tools.register("add")
        def add(a, b): return a + b

        @self.tools.register("multiply")
        def multiply(a, b): return a * b

    def _parse_command(self, text):
        # 手动解析命令
        if "+" in text:
            parts = text.split("+")
            return "add", float(parts[0]), float(parts[1])
        # ... 更多解析逻辑

    def run(self, query):
        prompt = f"Solve: {query}"
        response = self.llm.generate(prompt)
        tool_name, *args = self._parse_command(response)
        result = self.tools.execute(tool_name, *args)
        return result
```

#### LangChain (~10 行)

```python
from langchain.agents import initialize_agent, load_tools
from langchain.tools import Tool

# 定义工具
tools = [
    Tool(name="add", func=lambda a, b: a + b, description="Add two numbers"),
    Tool(name="multiply", func=lambda a, b: a * b, description="Multiply two numbers"),
]

# 创建 Agent
agent = initialize_agent(tools, llm, agent="zero-shot-react", verbose=True)

# 运行
result = agent.run("What is (5 + 3) * 2?")
```

---

## LangChain 示例

### 1. 基础 Agent

```python
from langchain.agents import initialize_agent, load_tools
from langchain import HuggingFacePipeline

# 创建 LLM
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
)

# 加载内置工具 (100+ 可用)
tools = load_tools(["llm-math", "wikipedia", "python_repl"], llm=llm)

# 创建 Agent (一行搞定!)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
)

# 使用
agent.run("Who is the current president of USA?")
```

### 2. ReAct Agent (推理 + 行动)

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

# 自定义工具
tools = [
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="Calculate math expressions"
    ),
    Tool(
        name="CodeRunner",
        func=lambda code: exec(code),
        description="Execute Python code"
    ),
]

# ReAct 提示
prompt = PromptTemplate.from_template("""
Answer questions using tools.

Tools: {tools}

Question: {input}
Thought: {agent_scratchpad}
""")

# 创建并运行
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
executor.invoke({"input": "Calculate 25% of 180"})
```

### 3. 内存对话 Agent

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 添加记忆功能
memory = ConversationBufferMemory()

# 创建对话链
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
)

# 多轮对话
conversation.predict(input="Hello, I'm John")
conversation.predict(input="What's my name?")  # 会记住 John
```

---

## LangGraph 示例

### 1. 基础工作流

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# 定义状态
class AgentState(TypedDict):
    input: str
    output: str
    messages: list

# 创建工作流
builder = StateGraph(AgentState)

# 添加节点
def process(state):
    result = llm(state["input"])
    return {"output": result, "messages": [result]}

builder.add_node("process", process)

# 设置入口和出口
builder.set_entry_point("process")
builder.add_edge("process", END)

# 编译并运行
graph = builder.compile()
result = graph.invoke({"input": "Hello", "messages": []})
```

### 2. 带条件路由的工作流

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

class CodeState(TypedDict):
    task: str
    code: str
    errors: Annotated[list, lambda x, y: x + y]
    done: bool

builder = StateGraph(CodeState)

# 添加节点
builder.add_node("generate", generate_code)
builder.add_node("test", test_code)
builder.add_node("fix", fix_code)

# 设置工作流
builder.set_entry_point("generate")
builder.add_edge("generate", "test")

# 条件路由
def check_result(state):
    return "fix" if state["errors"] else "done"

builder.add_conditional_edges(
    "test",
    check_result,
    {"fix": "fix", "done": END}
)
builder.add_edge("fix", "test")  # 循环

graph = builder.compile()
```

### 3. 多 Agent 系统

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class TeamState(TypedDict):
    task: str
    coder_output: str
    reviewer_output: str
    final: str

builder = StateGraph(TeamState)

# 多 Agent 协作
builder.add_node("coder", coder_agent)
builder.add_node("reviewer", reviewer_agent)
builder.add_node("finalizer", finalizer_agent)

builder.set_entry_point("coder")
builder.add_edge("coder", "reviewer")
builder.add_edge("reviewer", "finalizer")
builder.add_edge("finalizer", END)

graph = builder.compile()
```

---

## LangChain 内置工具

LangChain 提供 100+ 内置工具：

### 计算类
- `llm-math` - LLM 驱动的计算器
- `python_repl` - Python 代码执行
- `wolfram-alpha` - WolframAlpha 计算

### 搜索类
- `wikipedia` - Wikipedia 搜索
- `google-search` - Google 搜索
- `news` - 新闻搜索

### 开发类
- `terminal` - 终端命令执行
- `requests` - HTTP 请求
- `json` - JSON 处理

### 使用示例
```python
tools = load_tools([
    "llm-math",
    "wikipedia",
    "python_repl",
    "requests",
], llm=llm)
```

---

## 最佳实践

### 1. 选择合适的 Agent 类型

| Agent 类型 | 用途 | 复杂度 |
|-----------|------|--------|
| `zero-shot-react` | 通用任务 | 低 |
| `structured-chat` | 需要结构化输出 | 中 |
| `conversational` | 对话系统 | 低 |
| `custom (LangGraph)` | 复杂工作流 | 高 |

### 2. 工具设计原则

```python
# ✅ 好的工具设计
good_tool = Tool(
    name="calculator",
    func=calculate,
    description="Use for math calculations. Input: expression like '2+2'"
)

# ❌ 避免模糊的描述
bad_tool = Tool(
    name="tool1",
    func=some_func,
    description="Does stuff"  # 太模糊!
)
```

### 3. 错误处理

```python
from langchain.agents import AgentExecutor

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,  # 防止无限循环
    handle_parse_errors=True,  # 自动处理解析错误
    verbose=True,
)
```

### 4. 状态管理 (LangGraph)

```python
# ✅ 使用 Annotated 管理列表状态
class State(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]  # 追加
    errors: Annotated[list, lambda x, y: x + y]    # 追加
    counter: Annotated[int, operator.add]          # 累加

# ❌ 避免状态覆盖
class BadState(TypedDict):
    messages: list  # 每次都会覆盖!
```

---

## 运行示例

```bash
# LangChain 基础示例
python examples/langchain_agent.py basic

# ReAct Agent
python examples/langchain_agent.py react

# LangGraph 工作流
python examples/langgraph_agent.py code

# 查看工作流图
python examples/langgraph_agent.py diagram

# 运行所有示例
python examples/langchain_agent.py all
python examples/langgraph_agent.py all
```

---

## 总结

### LangChain 优势
- ✅ **10 倍代码减少**
- ✅ **100+ 内置工具**
- ✅ **多种 Agent 模式**
- ✅ **快速原型开发**

### LangGraph 优势
- ✅ **状态管理内置**
- ✅ **循环工作流支持**
- ✅ **多 Agent 编排**
- ✅ **可视化调试**

### 何时使用
- **从零构建**: 学习、高度定制需求
- **LangChain**: 快速原型、标准 Agent 模式
- **LangGraph**: 复杂工作流、多 Agent 系统

---

## 参考资源

- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [LangChain 工具列表](https://python.langchain.com/docs/integrations/tools/)
- 项目示例：`examples/langchain_agent.py`, `examples/langgraph_agent.py`
