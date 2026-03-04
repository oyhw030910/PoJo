"""Evaluation Script.

Evaluates a trained LLM agent on various environments.
"""

import os
import sys
import argparse
import json
from typing import Any, Dict, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from agent.policy import PolicyNetwork, PolicyConfig
from agent.llm_wrapper import LLMConfig
from environment.code_env import CodeEnvironment, CodeTask
from environment.math_env import MathEnvironment, MathTask
from evaluation.evaluator import Evaluator, EvaluatorConfig
from evaluation.metrics import MetricsCollector
from utils.helpers import seed_all
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate LLM agent")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config/eval_config.yaml",
                        help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                        help="Output directory")
    parser.add_argument("--environment", type=str, default="code",
                        choices=["code", "math", "gui"],
                        help="Environment to evaluate on")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--num-episodes", type=int, default=None,
                        help="Number of evaluation episodes")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic actions")
    parser.add_argument("--no-deterministic", action="store_false", dest="deterministic",
                        help="Use stochastic actions")

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def load_checkpoint(checkpoint_path: str, policy: PolicyNetwork, device: str) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Try loading directly
        policy.load_state_dict(checkpoint)

    return checkpoint.get("metrics", {}), checkpoint.get("config", {})


def create_code_tasks() -> list:
    """Create code evaluation tasks."""
    return [
        CodeTask(
            id="humaneval_0",
            description="Return the sum of two numbers.",
            starter_code="def add(a, b):\n    ",
            signature="def add(a, b)",
            test_cases=[
                {"args": [1, 2], "expected": 3},
                {"args": [10, 20], "expected": 30},
                {"args": [-5, 5], "expected": 0},
            ]
        ),
        CodeTask(
            id="humaneval_1",
            description="Return True if the number is prime, False otherwise.",
            starter_code="def is_prime(n):\n    ",
            signature="def is_prime(n)",
            test_cases=[
                {"args": [2], "expected": True},
                {"args": [17], "expected": True},
                {"args": [4], "expected": False},
                {"args": [1], "expected": False},
            ]
        ),
        CodeTask(
            id="humaneval_2",
            description="Return the factorial of n.",
            starter_code="def factorial(n):\n    ",
            signature="def factorial(n)",
            test_cases=[
                {"args": [5], "expected": 120},
                {"args": [0], "expected": 1},
                {"args": [1], "expected": 1},
            ]
        ),
    ]


def create_math_tasks() -> list:
    """Create math evaluation tasks."""
    return [
        MathTask(
            id="math_001",
            problem="If a train travels at 80 mph for 3 hours, how far does it travel?",
            solution="Distance = speed * time = 80 * 3 = 240 miles.",
            answer="240",
            topic="arithmetic",
        ),
        MathTask(
            id="math_002",
            problem="What is the value of x if 3x - 7 = 14?",
            solution="3x - 7 = 14. Add 7: 3x = 21. Divide by 3: x = 7.",
            answer="7",
            topic="algebra",
        ),
        MathTask(
            id="math_003",
            problem="A circle has radius 5. What is its area? (Use π = 3.14)",
            solution="Area = π * r² = 3.14 * 25 = 78.5",
            answer="78.5",
            topic="geometry",
        ),
        MathTask(
            id="math_004",
            problem="What is 35% of 80?",
            solution="35% of 80 = 0.35 * 80 = 28.",
            answer="28",
            topic="percentage",
        ),
    ]


def obs_to_tensor(obs: Any, device: str) -> Dict[str, torch.Tensor]:
    """Convert observation to tensor."""
    return {
        "input_ids": torch.zeros(1, 1, dtype=torch.long, device=device),
        "attention_mask": torch.ones(1, 1, dtype=torch.long, device=device),
    }


def action_to_tensor(action: Any) -> torch.Tensor:
    """Convert action to tensor."""
    if isinstance(action, torch.Tensor):
        return action
    return torch.tensor([0], dtype=torch.long)


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set seed
    seed_all(args.seed)

    # Setup logging
    logger = setup_logger("eval", log_dir=os.path.join(args.output_dir, "logs"))
    logger.info(f"Starting evaluation on {args.environment} environment")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {args.device}")

    # Load config
    config = load_config(args.config)

    # Create environment and tasks
    if args.environment == "code":
        env = CodeEnvironment({"max_steps": 30})
        tasks = create_code_tasks()
    elif args.environment == "math":
        env = MathEnvironment({"max_steps": 20})
        tasks = create_math_tasks()
    else:
        logger.error(f"Unknown environment: {args.environment}")
        return

    logger.info(f"Created {len(tasks)} evaluation tasks")

    # Create policy
    llm_config = LLMConfig(
        model_name=config.get("model", {}).get("name", "Qwen/Qwen2.5-1.5B-Instruct"),
        lora_enabled=False,  # Don't use LoRA for evaluation
    )

    policy_config = PolicyConfig(llm_config=llm_config)
    policy = PolicyNetwork(policy_config)

    # Load checkpoint
    metrics, train_config = load_checkpoint(args.checkpoint, policy, args.device)
    logger.info("Loaded checkpoint successfully")

    # Create evaluator config
    eval_config = EvaluatorConfig(
        num_episodes=args.num_episodes or config.get("evaluation", {}).get("num_episodes", 100),
        deterministic=args.deterministic,
        save_results=True,
        results_dir=args.output_dir,
    )

    # Create evaluator
    evaluator = Evaluator(
        policy=policy,
        env=env,
        config=eval_config,
        device=args.device,
        obs_to_tensor=lambda obs: obs_to_tensor(obs, args.device),
        action_to_tensor=action_to_tensor,
    )

    # Run evaluation
    logger.info("Starting evaluation...")
    result = evaluator.evaluate(verbose=True)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"eval_{args.environment}_{timestamp}.json")

    results_data = {
        "timestamp": timestamp,
        "checkpoint": args.checkpoint,
        "environment": args.environment,
        "num_episodes": result.num_episodes,
        "success_rate": result.success_rate,
        "avg_reward": result.avg_reward,
        "avg_episode_length": result.avg_episode_length,
        "metrics": result.metrics,
        "task_metrics": result.task_metrics,
        "training_metrics": metrics,
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Results saved to {results_file}")

    # Print summary
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Environment: {args.environment}")
    logger.info(f"Episodes: {result.num_episodes}")
    logger.info(f"Success Rate: {result.success_rate:.2%}")
    logger.info(f"Average Reward: {result.avg_reward:.3f}")
    logger.info(f"Average Episode Length: {result.avg_episode_length:.1f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
