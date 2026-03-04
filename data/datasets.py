"""Dataset Loader for RL-LLM Agent Training.

This module provides data loaders for common RL training datasets:
- Code: HumanEval, MBPP
- Math: GSM8K, MATH

Usage:
    from data.datasets import load_code_dataset, load_math_dataset

    code_tasks = load_code_dataset("humaneval")
    math_tasks = load_math_dataset("gsm8k")
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Code Generation Datasets
# ============================================================================

def load_code_dataset(name: str, data_dir: Optional[str] = None) -> List:
    """Load code generation dataset.

    Args:
        name: Dataset name ('humaneval', 'mbpp', 'demo')
        data_dir: Directory to load data from

    Returns:
        List of CodeTask objects
    """
    from environment.code_env import CodeTask

    if name.lower() == "humaneval":
        return _load_humaneval(data_dir)
    elif name.lower() == "mbpp":
        return _load_mbpp(data_dir)
    elif name.lower() == "demo":
        return _create_demo_code_tasks()
    else:
        raise ValueError(f"Unknown dataset: {name}")


def _load_humaneval(data_dir: Optional[str] = None) -> List:
    """Load HumanEval dataset.

    HumanEval contains 164 programming problems.
    https://huggingface.co/datasets/openai_humaneval
    """
    from environment.code_env import CodeTask

    if data_dir is None:
        # Return demo tasks if no data directory specified
        return _create_demo_code_tasks()

    data_path = os.path.join(data_dir, "humaneval.jsonl")

    if not os.path.exists(data_path):
        print(f"HumanEval data not found at {data_path}")
        print("Download from: https://huggingface.co/datasets/openai_humaneval")
        return _create_demo_code_tasks()

    tasks = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)

            # Parse test cases from canonical_solution
            task = CodeTask(
                id=f"humaneval_{item['task_id'].split('/')[-1]}",
                description=item['prompt'].strip(),
                starter_code=item['prompt'],
                signature="See prompt",
                test_cases=[]  # Would need to parse from test code
            )
            tasks.append(task)

    print(f"Loaded {len(tasks)} HumanEval tasks")
    return tasks


def _load_mbpp(data_dir: Optional[str] = None) -> List:
    """Load MBPP dataset.

    MBPP contains ~1000 crowd-sourced programming problems.
    https://huggingface.co/datasets/mbpp
    """
    from environment.code_env import CodeTask

    if data_dir is None:
        return _create_demo_code_tasks()

    data_path = os.path.join(data_dir, "mbpp.jsonl")

    if not os.path.exists(data_path):
        print(f"MBPP data not found at {data_path}")
        print("Download from: https://huggingface.co/datasets/mbpp")
        return _create_demo_code_tasks()

    tasks = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)

            task = CodeTask(
                id=f"mbpp_{item.get('task_id', len(tasks))}",
                description=item['text'],
                starter_code="",
                signature=item.get('code', '').split('(')[0].split()[-1] if item.get('code') else "solution",
                test_cases=[{"args": item.get('test_list', [])}]
            )
            tasks.append(task)

    print(f"Loaded {len(tasks)} MBPP tasks")
    return tasks


def _create_demo_code_tasks() -> List:
    """Create demo code tasks for quick testing."""
    from environment.code_env import CodeTask

    return [
        CodeTask(
            id="add_two_numbers",
            description="Write a function that adds two numbers together.",
            starter_code="def add(a, b):\n    ",
            signature="def add(a, b)",
            test_cases=[
                {"args": [1, 2], "expected": 3},
                {"args": [5, 7], "expected": 12},
                {"args": [-1, 1], "expected": 0},
                {"args": [0, 0], "expected": 0},
            ]
        ),
        CodeTask(
            id="is_even",
            description="Write a function that checks if a number is even.",
            starter_code="def is_even(n):\n    ",
            signature="def is_even(n)",
            test_cases=[
                {"args": [2], "expected": True},
                {"args": [3], "expected": False},
                {"args": [0], "expected": True},
                {"args": [-4], "expected": True},
            ]
        ),
        CodeTask(
            id="reverse_string",
            description="Write a function that reverses a string.",
            starter_code="def reverse(s):\n    ",
            signature="def reverse(s)",
            test_cases=[
                {"args": ["hello"], "expected": "olleh"},
                {"args": ["a"], "expected": "a"},
                {"args": [""], "expected": ""},
            ]
        ),
        CodeTask(
            id="factorial",
            description="Write a function that calculates factorial of n.",
            starter_code="def factorial(n):\n    ",
            signature="def factorial(n)",
            test_cases=[
                {"args": [0], "expected": 1},
                {"args": [1], "expected": 1},
                {"args": [5], "expected": 120},
                {"args": [10], "expected": 3628800},
            ]
        ),
        CodeTask(
            id="is_prime",
            description="Write a function that checks if a number is prime.",
            starter_code="def is_prime(n):\n    ",
            signature="def is_prime(n)",
            test_cases=[
                {"args": [2], "expected": True},
                {"args": [17], "expected": True},
                {"args": [4], "expected": False},
                {"args": [1], "expected": False},
            ]
        ),
    ]


# ============================================================================
# Math Datasets
# ============================================================================

def load_math_dataset(name: str, data_dir: Optional[str] = None) -> List:
    """Load math reasoning dataset.

    Args:
        name: Dataset name ('gsm8k', 'math', 'demo')
        data_dir: Directory to load data from

    Returns:
        List of MathTask objects
    """
    from environment.math_env import MathTask

    if name.lower() == "gsm8k":
        return _load_gsm8k(data_dir)
    elif name.lower() == "math":
        return _load_math_dataset(data_dir)
    elif name.lower() == "demo":
        return _create_demo_math_tasks()
    else:
        raise ValueError(f"Unknown dataset: {name}")


def _load_gsm8k(data_dir: Optional[str] = None) -> List:
    """Load GSM8K dataset.

    GSM8K contains grade school math word problems.
    https://huggingface.co/datasets/gsm8k
    """
    from environment.math_env import MathTask

    if data_dir is None:
        return _create_demo_math_tasks()

    data_path = os.path.join(data_dir, "gsm8k.jsonl")

    if not os.path.exists(data_path):
        print(f"GSM8K data not found at {data_path}")
        print("Download from: https://huggingface.co/datasets/gsm8k")
        return _create_demo_math_tasks()

    tasks = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)

            task = MathTask(
                id=f"gsm8k_{len(tasks)}",
                problem=item['question'],
                solution=item['answer'],
                answer=extract_number(item['answer']),
                topic="arithmetic"
            )
            tasks.append(task)

    print(f"Loaded {len(tasks)} GSM8K tasks")
    return tasks


def _load_math_dataset(data_dir: Optional[str] = None) -> List:
    """Load MATH dataset.

    MATH contains challenging competition mathematics problems.
    https://huggingface.co/datasets/hendrycksMATH
    """
    from environment.math_env import MathTask

    if data_dir is None:
        return _create_demo_math_tasks()

    data_path = os.path.join(data_dir, "math.jsonl")

    if not os.path.exists(data_path):
        print(f"MATH data not found at {data_path}")
        print("Download from: https://huggingface.co/datasets/hendrycksMATH")
        return _create_demo_math_tasks()

    tasks = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)

            task = MathTask(
                id=f"math_{len(tasks)}",
                problem=item['problem'],
                solution=item['solution'],
                answer=item.get('answer', ''),
                topic=item.get('topic', 'general')
            )
            tasks.append(task)

    print(f"Loaded {len(tasks)} MATH tasks")
    return tasks


def extract_number(text: str) -> str:
    """Extract numerical answer from text."""
    import re
    # Look for patterns like "#### 42" or answer markers
    match = re.search(r'####\s*(\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1)

    # Try to find any number
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]

    return text.strip()


def _create_demo_math_tasks() -> List:
    """Create demo math tasks for quick testing."""
    from environment.math_env import MathTask

    return [
        MathTask(
            id="demo_arithmetic_1",
            problem="What is 25 + 17?",
            solution="25 + 17 = 42",
            answer="42",
            topic="arithmetic"
        ),
        MathTask(
            id="demo_percentage_1",
            problem="What is 15% of 80?",
            solution="15% of 80 = 0.15 × 80 = 12",
            answer="12",
            topic="percentage"
        ),
        MathTask(
            id="demo_word_1",
            problem="John has 10 apples. He gives 3 to Mary and buys 5 more. How many apples does he have now?",
            solution="10 - 3 + 5 = 12",
            answer="12",
            topic="arithmetic"
        ),
        MathTask(
            id="demo_algebra_1",
            problem="If 2x + 5 = 15, what is x?",
            solution="2x + 5 = 15\n2x = 10\nx = 5",
            answer="5",
            topic="algebra"
        ),
        MathTask(
            id="demo_geometry_1",
            problem="What is the area of a rectangle with length 8 and width 5?",
            solution="Area = length × width = 8 × 5 = 40",
            answer="40",
            topic="geometry"
        ),
    ]


# ============================================================================
# Dataset Info
# ============================================================================

DATASET_INFO = {
    "humaneval": {
        "description": "HumanEval: 164 hand-written programming problems",
        "url": "https://huggingface.co/datasets/openai_humaneval",
        "format": "JSONL with task_id, prompt, canonical_solution, test",
        "size": "164 tasks"
    },
    "mbpp": {
        "description": "MBPP: ~1000 crowd-sourced Python problems",
        "url": "https://huggingface.co/datasets/mbpp",
        "format": "JSONL with task_id, text, code, test_list",
        "size": "~1000 tasks"
    },
    "gsm8k": {
        "description": "GSM8K: Grade school math word problems",
        "url": "https://huggingface.co/datasets/gsm8k",
        "format": "JSONL with question, answer",
        "size": "8.5K train, 1.3K test"
    },
    "math": {
        "description": "MATH: Competition mathematics problems",
        "url": "https://huggingface.co/datasets/hendrycksMATH",
        "format": "JSONL with problem, solution, answer, topic",
        "size": "12.5K problems"
    }
}


def list_datasets():
    """List available datasets with info."""
    print("Available Datasets:")
    print("=" * 60)
    for name, info in DATASET_INFO.items():
        print(f"\n{name.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Size: {info['size']}")
        print(f"  URL: {info['url']}")


if __name__ == "__main__":
    list_datasets()

    # Test loading
    print("\n" + "=" * 60)
    print("Testing demo datasets:")

    code_tasks = load_code_dataset("demo")
    print(f"\nCode demo tasks: {len(code_tasks)}")

    math_tasks = load_math_dataset("demo")
    print(f"Math demo tasks: {len(math_tasks)}")
