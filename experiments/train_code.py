"""Training Script for Code Environment.

Trains an LLM agent on code generation tasks using PPO.
"""

import os
import sys
import argparse
from typing import Any, Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from tqdm import tqdm

from agent.policy import PolicyNetwork, PolicyConfig
from agent.llm_wrapper import LLMConfig
from environment.code_env import CodeEnvironment, CodeTask
from rl.ppo_trainer import PPOTrainer, PPOConfig
from rl.trainer import RLTrainer, TrainingConfig
from utils.helpers import seed_all
from utils.logger import setup_logger, TrainingLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LLM agent on code tasks")

    parser.add_argument("--config", type=str, default="config/train_config.yaml",
                        help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Override arguments
    parser.add_argument("--model", type=str, default=None,
                        help="Model name or path")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Total training steps")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def create_sample_tasks() -> list:
    """Create sample code tasks for training."""
    tasks = [
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
            id="count_vowels",
            description="Write a function that counts the number of vowels in a string.",
            starter_code="def count_vowels(s):\n    ",
            signature="def count_vowels(s)",
            test_cases=[
                {"args": ["hello"], "expected": 2},
                {"args": ["AEIOU"], "expected": 5},
                {"args": ["xyz"], "expected": 0},
            ]
        ),
    ]
    return tasks


def obs_to_tensor(obs: Any, device: str) -> Dict[str, torch.Tensor]:
    """Convert observation to tensor format for policy."""
    # Get text from observation
    if hasattr(obs, 'text'):
        text = obs.text
    else:
        text = str(obs)

    # This would normally use the tokenizer from the policy's LLM
    # For now, return a placeholder
    return {
        "input_ids": torch.zeros(1, 1, dtype=torch.long, device=device),
        "attention_mask": torch.ones(1, 1, dtype=torch.long, device=device),
    }


def action_to_tensor(action: Any) -> torch.Tensor:
    """Convert action to tensor format."""
    if isinstance(action, torch.Tensor):
        return action
    if isinstance(action, (int, float)):
        return torch.tensor([action], dtype=torch.long)
    # For text actions, we'd tokenize
    return torch.zeros(1, dtype=torch.long)


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    seed_all(args.seed)

    # Setup logging
    logger = setup_logger("train_code", log_dir=os.path.join(args.output_dir, "logs"))
    training_logger = TrainingLogger(log_dir=args.output_dir)

    logger.info("Starting code environment training")
    logger.info(f"Device: {args.device}")
    logger.info(f"Seed: {args.seed}")

    # Load config
    config = load_config(args.config)

    # Override with command line args
    if args.model:
        config.setdefault("model", {})["name"] = args.model
    if args.total_steps:
        config.setdefault("training", {})["total_steps"] = args.total_steps
    if args.lr:
        config.setdefault("ppo", {})["lr"] = args.lr

    # Create environment
    env = CodeEnvironment(config.get("environment", {}))
    tasks = create_sample_tasks()
    logger.info(f"Created {len(tasks)} sample tasks")

    # Create policy
    llm_config = LLMConfig(
        model_name=config.get("model", {}).get("name", "Qwen/Qwen2.5-1.5B-Instruct"),
        lora_enabled=config.get("model", {}).get("lora", {}).get("enabled", True),
        lora_r=config.get("model", {}).get("lora", {}).get("r", 32),
        lora_alpha=config.get("model", {}).get("lora", {}).get("alpha", 64),
    )

    policy_config = PolicyConfig(
        llm_config=llm_config,
        use_value_head=True,
    )

    logger.info("Creating policy network...")
    policy = PolicyNetwork(policy_config)
    logger.info(f"Policy created with {sum(p.numel() for p in policy.parameters()):,} parameters")

    # Create PPO trainer
    ppo_config = PPOConfig(
        lr=config.get("ppo", {}).get("lr", 3e-5),
        epochs=config.get("ppo", {}).get("epochs", 4),
        batch_size=config.get("ppo", {}).get("batch_size", 32),
        mini_batch_size=config.get("ppo", {}).get("mini_batch_size", 4),
        gamma=config.get("ppo", {}).get("gamma", 0.99),
        lam=config.get("ppo", {}).get("lam", 0.95),
        clip_epsilon=config.get("ppo", {}).get("clip_epsilon", 0.2),
    )

    # Create training config
    training_config = TrainingConfig(
        algorithm="ppo",
        total_steps=config.get("training", {}).get("total_steps", 10000),
        rollout_steps=config.get("training", {}).get("rollout_steps", 128),
        eval_interval=config.get("training", {}).get("eval_interval", 500),
        save_interval=config.get("training", {}).get("save_interval", 1000),
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        log_dir=os.path.join(args.output_dir, "logs"),
    )

    # Create RL trainer
    trainer = RLTrainer(
        policy=policy,
        env=env,
        config=training_config,
        ppo_config=ppo_config,
        device=args.device,
        obs_to_tensor=lambda obs: obs_to_tensor(obs, args.device),
        action_to_tensor=action_to_tensor,
    )

    logger.info("Starting training...")

    # Training loop
    task_idx = 0
    for step in range(training_config.total_steps):
        # Rotate through tasks
        task = tasks[task_idx % len(tasks)]
        task_idx += 1

        # Reset environment with task
        obs = env.reset(task=task)

        # Collect rollout
        rollout = trainer.collect_rollout(num_steps=training_config.rollout_steps)

        # Update policy
        stats = trainer.update_policy(rollout)

        # Log metrics
        avg_reward = sum(rollout.rewards) / len(rollout.rewards)

        if step % training_config.log_interval == 0:
            training_logger.log_scalar("loss/policy", stats.policy_loss, step)
            training_logger.log_scalar("loss/value", stats.value_loss, step)
            training_logger.log_scalar("metrics/entropy", stats.entropy, step)
            training_logger.log_scalar("metrics/reward", avg_reward, step)

            logger.info(f"Step {step}: Policy Loss={stats.policy_loss:.4f}, "
                       f"Value Loss={stats.value_loss:.4f}, Avg Reward={avg_reward:.3f}")

        # Evaluation
        if step % training_config.eval_interval == 0:
            eval_metrics = trainer.evaluate(num_episodes=10)
            training_logger.log_metrics(eval_metrics, step)
            logger.info(f"Eval at step {step}: {eval_metrics}")

        # Save checkpoint
        if step % training_config.save_interval == 0:
            checkpoint_path = os.path.join(
                training_config.checkpoint_dir,
                f"checkpoint_step_{step}.pt"
            )
            trainer.save_checkpoint(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Final save
    final_path = os.path.join(training_config.checkpoint_dir, "final_model.pt")
    trainer.save_checkpoint(final_path)
    logger.info(f"Saved final model to {final_path}")

    training_logger.close()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
