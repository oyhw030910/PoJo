# RL-LLM Agent 训练数据指南

本目录说明训练数据的来源、加载方法和使用方式。

## 目录结构

```
data/
├── __init__.py           # 数据模块导出
├── datasets.py           # 数据集加载器
└── README.md             # 本文档
```

## 支持的数据集

### 代码生成数据集

| 数据集 | 描述 | 任务数 | 加载方法 |
|--------|------|--------|----------|
| **HumanEval** | 手写编程问题 | 164 | `load_code_dataset("humaneval")` |
| **MBPP** | 众包 Python 问题 | ~1000 | `load_code_dataset("mbpp")` |
| **Demo** | 示例任务 | 5 | `load_code_dataset("demo")` |

### 数学推理数据集

| 数据集 | 描述 | 任务数 | 加载方法 |
|--------|------|--------|----------|
| **GSM8K** | 小学应用题 | 8.5K | `load_math_dataset("gsm8k")` |
| **MATH** | 竞赛数学题 | 12.5K | `load_math_dataset("math")` |
| **Demo** | 示例任务 | 5 | `load_math_dataset("demo")` |

---

## 快速开始

### 1. 使用 Demo 数据（无需下载）

```python
from data.datasets import load_code_dataset, load_math_dataset

# 加载代码数据
code_tasks = load_code_dataset("demo")
print(f"Loaded {len(code_tasks)} code tasks")

# 加载数学数据
math_tasks = load_math_dataset("demo")
print(f"Loaded {len(math_tasks)} math tasks")
```

### 2. 使用真实数据集

#### 下载 HumanEval

```bash
# 从 HuggingFace 下载
git lfs install
git clone https://huggingface.co/datasets/openai_humaneval
mv humaneval data/humaneval
```

#### 下载 GSM8K

```bash
# 从 HuggingFace 下载
git lfs install
git clone https://huggingface.co/datasets/gsm8k
mv gsm8k/data data/gsm8k
```

#### 加载数据

```python
from data.datasets import load_code_dataset, load_math_dataset

# 加载 HumanEval
code_tasks = load_code_dataset("humaneval", data_dir="./data")

# 加载 GSM8K
math_tasks = load_math_dataset("gsm8k", data_dir="./data")
```

---

## 训练脚本使用

### 代码环境训练

```bash
# 使用 Demo 数据训练
python experiments/train_code.py --dataset demo

# 使用 HumanEval 训练
python experiments/train_code.py --dataset humaneval --data-dir ./data

# 使用 MBPP 训练
python experiments/train_code.py --dataset mbpp --data-dir ./data
```

### 数学环境训练

```bash
# 使用 Demo 数据训练
python experiments/train_math.py --dataset demo

# 使用 GSM8K 训练
python experiments/train_math.py --dataset gsm8k --data-dir ./data
```

---

## 数据格式

### HumanEval 格式 (JSONL)

```json
{
  "task_id": "HumanEval/0",
  "prompt": "def plus_one(digits: List[int]) -> List[int]:",
  "canonical_solution": "    result = []\n    carry = 1...",
  "test": "def check(candidate):\n    assert candidate([1, 2, 3]) == [1, 2, 4]"
}
```

### MBPP 格式 (JSONL)

```json
{
  "task_id": 1,
  "text": "Write a function to add two numbers",
  "code": "def add(a, b):\n    return a + b",
  "test_list": ["add(2, 3) == 5", "add(-1, 1) == 0"]
}
```

### GSM8K 格式 (JSONL)

```json
{
  "question": "John has 10 apples. He gives 3 to Mary...",
  "answer": "John starts with 10 apples. After giving 3, he has 10 - 3 = 7..."
}
```

---

## 自定义数据集

### 创建 CodeTask

```python
from environment.code_env import CodeTask

task = CodeTask(
    id="my_task",
    description="Write a function to...",
    starter_code="def solution(x):\n    ",
    signature="def solution(x)",
    test_cases=[
        {"args": [input1], "expected": output1},
        {"args": [input2], "expected": output2},
    ]
)
```

### 创建 MathTask

```python
from environment.math_env import MathTask

task = MathTask(
    id="my_math_task",
    problem="What is 25% of 80?",
    solution="25% of 80 = 0.25 × 80 = 20",
    answer="20",
    topic="percentage"
)
```

---

## 数据存储位置

```
rl_llm_agent/
├── data/                    # 数据加载模块
│   ├── datasets.py         # 数据集加载器
│   └── README.md           # 本文档
├── experiments/             # 训练脚本
│   ├── train_code.py
│   └── train_math.py
└── [data_dir]/             # 数据存放目录（需创建）
    ├── humaneval/          # HumanEval 数据
    ├── mbpp/               # MBPP 数据
    └── gsm8k/              # GSM8K 数据
```

---

## 数据集下载链接

| 数据集 | HuggingFace 链接 |
|--------|-----------------|
| HumanEval | https://huggingface.co/datasets/openai_humaneval |
| MBPP | https://huggingface.co/datasets/mbpp |
| GSM8K | https://huggingface.co/datasets/gsm8k |
| MATH | https://huggingface.co/datasets/hendrycksMATH |

---

## 当前状态

| 数据集 | 状态 | 说明 |
|--------|------|------|
| Demo | ✅ 内置 | 无需下载，直接可用 |
| HumanEval | ⚠️ 需下载 | 从 HuggingFace 获取 |
| MBPP | ⚠️ 需下载 | 从 HuggingFace 获取 |
| GSM8K | ⚠️ 需下载 | 从 HuggingFace 获取 |
| MATH | ⚠️ 需下载 | 从 HuggingFace 获取 |

---

## 常见问题

### Q: 为什么只显示 Demo 数据？

A: 真实数据集需要从 HuggingFace 下载。设置 `--data-dir` 参数指定数据目录即可。

### Q: 数据格式不匹配怎么办？

A: 可以修改 `data/datasets.py` 中的加载函数来适配不同的数据格式。

### Q: 如何添加自己的数据集？

A: 参考 `data/datasets.py` 中的示例，创建新的加载函数并返回 `CodeTask` 或`MathTask` 列表。

---

## 参考资源

- [HumanEval 论文](https://arxiv.org/abs/2107.03374)
- [MBPP 论文](https://arxiv.org/abs/2108.07732)
- [GSM8K 论文](https://arxiv.org/abs/2110.14168)
- [MATH 论文](https://arxiv.org/abs/2103.03874)
