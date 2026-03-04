"""Training Script for Math Environment.

Trains an LLM agent on mathematical reasoning tasks using PPO or GRPO.
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
from environment.math_env import MathEnvironment, MathTask
from rl.ppo_trainer import PPOTrainer, PPOConfig
from rl.grpo_trainer import GRPOTrainer, GRPOConfig
from rl.trainer import RLTrainer, TrainingConfig
from utils.helpers import seed_all
from utils.logger import setup_logger, TrainingLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LLM agent on math tasks")

    parser.add_argument("--config", type=str, default="config/train_config.yaml",
                        help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="./outputs_math",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--algorithm", type=str, default="ppo",
                        choices=["ppo", "grpo"],
                        help="RL algorithm to use")

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
    """Create sample math tasks for training."""
    tasks = [
        MathTask(
            id="gsm8k_001",
            problem="John has 5 apples. He buys 3 more apples. How many apples does he have now?",
            solution="John starts with 5 apples. He buys 3 more. 5 + 3 = 8. John has 8 apples.",
            answer="8",
            topic="arithmetic",
        ),
        MathTask(
            id="gsm8k_002",
            problem="A shirt costs $25. It is on sale for 20% off. What is the sale price?",
            solution="The discount is 20% of $25 = 0.20 * 25 = $5. Sale price = $25 - $5 = $20.",
            answer="20",
            topic="percentage",
        ),
        MathTask(
            id="gsm8k_003",
            problem="If 2x + 3 = 11, what is the value of x?",
            solution="2x + 3 = 11. Subtract 3 from both sides: 2x = 8. Divide by 2: x = 4.",
            answer="4",
            topic="algebra",
        ),
        MathTask(
            id="gsm8k_004",
            problem="What is the area of a rectangle with length 8 cm and width 5 cm?",
            solution="Area = length * width = 8 * 5 = 40 square cm.",
            answer="40",
            topic="geometry",
        ),
        MathTask(
            id="gsm8k_005",
            problem="Sarah runs 3 miles per day for 5 days. How many miles does she run in total?",
            solution="Sarah runs 3 miles/day * 5 days = 15 miles total.",
            answer="15",
            topic="arithmetic",
        ),
        MathTask(
            id="gsm8k_006",
            problem="What is 15% of 200?",
            solution="15% of 200 = 0.15 * 200 = 30.",
            answer="30",
            topic="percentage",
        ),
        MathTask(
            id="gsm8k_007",
            problem="The sum of two consecutive numbers is 25. What is the smaller number?",
            solution="Let the numbers be n and n+1. n + (n+1) = 25. 2n + 1 = 25. 2n = 24. n = 12.",
            answer="12",
            topic="algebra",
        ),
        MathTask(
            id="gsm8k_008",
            problem="A car travels 60 miles per hour. How far does it travel in 2.5 hours?",
            solution="Distance = speed * time = 60 * 2.5 = 150 miles.",
            answer="150",
            topic="arithmetic",
        ),
    ]
    return tasks


def obs_to_tensor(obs: Any, device: str) -> Dict[str, torch.Tensor]:
    """Convert observation to tensor format for policy."""
    if hasattr(obs, 'text'):
        text = obs.text
    else:
        text = str(obs)

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
    return torch.zeros(1, dtype=torch.long)


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    seed_all(args.seed)

    # Setup logging
    logger = setup_logger("train_math", log_dir=os.path.join(args.output_dir, "logs"))
    training_logger = TrainingLogger(log_dir=args.output_dir)

    logger.info("Starting math environment training")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Device: {args.device}")

    # Load config
    config = load_config(args.config)

    # Override with command line args
    if args.model:
        config.setdefault("model", {})["name"] = args.model
    if args.total_steps:
        config.setdefault("training", {})["total_steps"] = args.total_steps
    if args.lr:
        config.setdefault("ppo", {})["lr"] = args.lr
        config.setdefault("grpo", {})["lr"] = args.lr

    # Create environment
    env = MathEnvironment(config.get("environment", {}))
    tasks = create_sample_tasks()
    logger.info(f"Created {len(tasks)} sample tasks")

    # Create policy
    llm_config = LLMConfig(
        model_name=config.get("model", {}).get("name", "Qwen/Qwen2.5-1.5B-Instruct"),
        lora_enabled=config.get("model", {}).get("lora", {}).get("enabled", True),
        lora_r=config.get("model", {}).get("lora", {}).get("r", 32),
        lora_alpha=config.get("model", {}).get("lora", {}).get("alpha", 64),
    )

    policy_config = PolicyConfig(llm_config=llm_config, use_value_head=(args.algorithm == "ppo"))

    logger.info("Creating policy network...")
    policy = PolicyNetwork(policy_config)
    logger.info(f"Policy created with {sum(p.numel() for p in policy.parameters()):,} parameters")

    # Create trainer based on algorithm
    if args.algorithm == "ppo":
        algo_config = PPOConfig(
            lr=config.get("ppo", {}).get("lr", 3e-5),
            epochs=config.get("ppo", {}).get("epochs", 4),
            batch_size=config.get("ppo", {}).get("batch_size", 32),
            mini_batch_size=config.get("ppo", {}).get("mini_batch_size", 4),
            gamma=config.get("ppo", {}).get("gamma", 0.99),
            lam=config.get("ppo", {}).get("lam", 0.95),
        )
        logger.info("Using PPO algorithm")
    else:
        algo_config = GRPOConfig(
            lr=config.get("grpo", {}).get("lr", 3e-5),
            epochs=config.get("grpo", {}).get("epochs", 4),
            batch_size=config.get("grpo", {}).get("batch_size", 32),
            mini_batch_size=config.get("grpo", {}).get("mini_batch_size", 4),
            group_size=config.get("grpo", {}).get("group_size", 8),
        )
        logger.info("Using GRPO algorithm")

    # Create training config
    training_config = TrainingConfig(
        algorithm=args.algorithm,
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
        ppo_config=algo_config if args.algorithm == "ppo" else None,
        grpo_config=algo_config if args.algorithm == "grpo" else None,
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
            training_logger.log_scalar("metrics/entropy", stats.entropy, step)
            training_logger.log_scalar("metrics/reward", avg_reward, step)

            logger.info(f"Step {step}: Policy Loss={stats.policy_loss:.4f}, Avg Reward={avg_reward:.3f}")

        # Evaluation
        if step % training_config.eval_interval == 0:
            eval_metrics = trainer.evaluate(num_episodes=20)
            training_logger.log_metrics(eval_metrics, step)
            logger.info(f"Eval at step {step}: Success Rate={eval_metrics.get('eval_success_rate', 0):.2%}")

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
