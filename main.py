"""Main entry point for RL-LLM Agent."""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RL-LLM Agent Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on code environment
  python -m rl_llm_agent train --env code

  # Train on math environment with GRPO
  python -m rl_llm_agent train --env math --algorithm grpo

  # Evaluate a model
  python -m rl_llm_agent eval --checkpoint ./checkpoints/model.pt

  # Run interactive demo
  python -m rl_llm_agent demo --env code
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--env", type=str, default="code",
                              choices=["code", "math", "gui"],
                              help="Environment to train on")
    train_parser.add_argument("--algorithm", type=str, default="ppo",
                              choices=["ppo", "grpo"],
                              help="RL algorithm")
    train_parser.add_argument("--config", type=str, default="config/train_config.yaml",
                              help="Path to config file")
    train_parser.add_argument("--output-dir", type=str, default="./outputs",
                              help="Output directory")
    train_parser.add_argument("--model", type=str, default=None,
                              help="Model name or path")
    train_parser.add_argument("--total-steps", type=int, default=None,
                              help="Total training steps")
    train_parser.add_argument("--lr", type=float, default=None,
                              help="Learning rate")
    train_parser.add_argument("--seed", type=int, default=42,
                              help="Random seed")
    train_parser.add_argument("--device", type=str, default="cuda",
                              help="Device to use")
    train_parser.add_argument("--resume", type=str, default=None,
                              help="Path to checkpoint to resume from")

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument("--checkpoint", type=str, required=True,
                             help="Path to model checkpoint")
    eval_parser.add_argument("--env", type=str, default="code",
                             choices=["code", "math", "gui"],
                             help="Environment to evaluate on")
    eval_parser.add_argument("--config", type=str, default="config/eval_config.yaml",
                             help="Path to config file")
    eval_parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                             help="Output directory")
    eval_parser.add_argument("--num-episodes", type=int, default=100,
                             help="Number of evaluation episodes")
    eval_parser.add_argument("--deterministic", action="store_true", default=True,
                             help="Use deterministic actions")
    eval_parser.add_argument("--device", type=str, default="cuda",
                             help="Device to use")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")
    demo_parser.add_argument("--env", type=str, default="code",
                             choices=["code", "math", "gui"],
                             help="Environment for demo")
    demo_parser.add_argument("--model", type=str, default=None,
                             help="Model checkpoint for demo")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    if args.command == "train":
        from experiments.train_code import main as train_code
        from experiments.train_math import main as train_math

        if args.env == "code":
            train_code()
        elif args.env == "math":
            train_math()
        else:
            print("GUI training not yet supported")
            sys.exit(1)

    elif args.command == "eval":
        from experiments.eval import main as eval_main
        eval_main()

    elif args.command == "demo":
        run_demo(args)

    else:
        print("No command specified. Use --help for usage.")
        sys.exit(1)


def run_demo(args):
    """Run interactive demo."""
    print(f"Running demo on {args.env} environment...")

    # Import required modules
    from agent.policy import PolicyNetwork, PolicyConfig
    from agent.llm_wrapper import LLMConfig
    from environment.code_env import CodeEnvironment
    from environment.math_env import MathEnvironment

    # Create environment
    if args.env == "code":
        env = CodeEnvironment({"max_steps": 10})
        print("\nCode Environment Demo")
        print("=" * 40)
    elif args.env == "math":
        env = MathEnvironment({"max_steps": 10})
        print("\nMath Environment Demo")
        print("=" * 40)
    else:
        print("GUI demo not yet supported")
        return

    # Print task description
    print(env.get_task_description())
    print("\nCommands:")
    print("  - Enter your action (e.g., 'generate: def foo(): ...')")
    print("  - 'quit' to exit")
    print("  - 'reset' to restart")
    print("=" * 40)

    obs = env.reset()
    print(f"\nObservation:\n{obs.text}")

    while True:
        try:
            user_input = input("\n> ").strip()

            if user_input.lower() == "quit":
                print("Exiting demo.")
                break

            if user_input.lower() == "reset":
                obs = env.reset()
                print(f"\nReset. New observation:\n{obs.text}")
                continue

            # Parse action
            if ":" in user_input:
                action_type, value = user_input.split(":", 1)
                action_type = action_type.strip()
                value = value.strip()
            else:
                action_type = user_input
                value = None

            from environment.base_env import Action
            action = Action(action_type=action_type, value=value)

            # Check if valid
            if not env.is_valid_action(action):
                print(f"Invalid action. Valid types: {env.get_action_space().action_types}")
                continue

            # Execute
            result = env.step(action)
            print(f"\nReward: {result.reward:.3f}")
            print(f"Done: {result.done}")
            print(f"Info: {result.info}")
            print(f"\nObservation:\n{result.observation.text}")

            if result.done:
                print("\nEpisode complete! Enter 'reset' to start new episode.")

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
