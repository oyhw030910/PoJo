"""IO Utilities Module.

Provides file IO utilities for RL-LLM Agent.
"""

import os
import json
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import yaml


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    filepath: str = "checkpoint.pt"
) -> None:
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optional optimizer
        scheduler: Optional scheduler
        metrics: Optional metrics dictionary
        config: Optional configuration
        filepath: Output file path
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if config is not None:
        checkpoint["config"] = config

    # Create directory if needed
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu"
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        filepath: Checkpoint file path
        model: Model to load weights into
        optimizer: Optional optimizer to load into
        scheduler: Optional scheduler to load into
        device: Target device

    Returns:
        Additional checkpoint data
    """
    checkpoint = torch.load(filepath, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
        "timestamp": checkpoint.get("timestamp", "unknown"),
    }


def save_model(
    model: torch.nn.Module,
    filepath: str,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Save model weights.

    Args:
        model: Model to save
        filepath: Output file path
        config: Optional configuration
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    save_data = {
        "model_state_dict": model.state_dict(),
    }

    if config is not None:
        save_data["config"] = config

    torch.save(save_data, filepath)


def load_model(
    filepath: str,
    model: torch.nn.Module,
    device: Union[str, torch.device] = "cpu"
) -> Dict[str, Any]:
    """Load model weights.

    Args:
        filepath: Model file path
        model: Model to load weights into
        device: Target device

    Returns:
        Additional model data
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return {
        "config": checkpoint.get("config", {}),
    }


def save_results(
    results: Dict[str, Any],
    filepath: str,
    format: str = "json"
) -> str:
    """Save evaluation results.

    Args:
        results: Results dictionary
        filepath: Output file path
        format: Output format ('json' or 'yaml')

    Returns:
        Path to saved file
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    if format == "yaml":
        with open(filepath, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    else:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    return filepath


def load_results(
    filepath: str,
    format: Optional[str] = None
) -> Dict[str, Any]:
    """Load evaluation results.

    Args:
        filepath: Results file path
        format: File format (auto-detected if None)

    Returns:
        Results dictionary
    """
    if format is None:
        format = "yaml" if filepath.endswith(".yaml") or filepath.endswith(".yml") else "json"

    if format == "yaml":
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    else:
        with open(filepath, 'r') as f:
            return json.load(f)


def save_config(
    config: Dict[str, Any],
    filepath: str,
    format: str = "yaml"
) -> None:
    """Save configuration.

    Args:
        config: Configuration dictionary
        filepath: Output file path
        format: Output format
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    if format == "yaml":
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)


def load_config(
    filepath: str
) -> Dict[str, Any]:
    """Load configuration.

    Args:
        filepath: Configuration file path

    Returns:
        Configuration dictionary
    """
    if filepath.endswith(".yaml") or filepath.endswith(".yml"):
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    else:
        with open(filepath, 'r') as f:
            return json.load(f)


def list_checkpoints(
    checkpoint_dir: str
) -> List[str]:
    """List all checkpoints in directory.

    Args:
        checkpoint_dir: Directory to search

    Returns:
        List of checkpoint paths
    """
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(".pt") or filename.endswith(".pth"):
            checkpoints.append(os.path.join(checkpoint_dir, filename))
    return sorted(checkpoints)


def get_latest_checkpoint(
    checkpoint_dir: str
) -> Optional[str]:
    """Get the most recent checkpoint.

    Args:
        checkpoint_dir: Directory to search

    Returns:
        Path to latest checkpoint or None
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    return checkpoints[-1] if checkpoints else None


def export_model_for_inference(
    model: torch.nn.Module,
    filepath: str,
    tokenizer: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Export model for inference (without training overhead).

    Args:
        model: Model to export
        filepath: Output directory
        tokenizer: Optional tokenizer to save
        config: Optional configuration
    """
    os.makedirs(filepath, exist_ok=True)

    # Save model (without optimizer state)
    model_to_save = getattr(model, 'module', model)  # Handle DDP wrapper
    model_to_save.save_pretrained(filepath)

    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(filepath)

    # Save config
    if config is not None:
        save_config(config, os.path.join(filepath, "rl_config.yaml"))


def import_model_from_pretrained(
    filepath: str,
    model_class: type,
    device: Union[str, torch.device] = "cpu"
) -> torch.nn.Module:
    """Import model from pretrained checkpoint.

    Args:
        filepath: Checkpoint directory or file
        model_class: Model class to instantiate
        device: Target device

    Returns:
        Loaded model
    """
    # Create model instance
    model = model_class()
    model.to(device)

    # Load weights
    if os.path.isdir(filepath):
        # Try to find checkpoint file
        ckpt_file = os.path.join(filepath, "pytorch_model.bin")
        if not os.path.exists(ckpt_file):
            ckpt_file = os.path.join(filepath, "model.safetensors")
    else:
        ckpt_file = filepath

    checkpoint = torch.load(ckpt_file, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def save_trajectories(
    trajectories: List[Dict[str, Any]],
    filepath: str,
    compress: bool = False
) -> str:
    """Save training trajectories.

    Args:
        trajectories: List of trajectory dictionaries
        filepath: Output file path
        compress: Whether to compress

    Returns:
        Path to saved file
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    if compress:
        import gzip
        with gzip.open(filepath + ".gz", 'wt') as f:
            json.dump(trajectories, f)
        return filepath + ".gz"
    else:
        with open(filepath, 'w') as f:
            json.dump(trajectories, f, indent=2)
        return filepath


def load_trajectories(
    filepath: str
) -> List[Dict[str, Any]]:
    """Load training trajectories.

    Args:
        filepath: Trajectory file path

    Returns:
        List of trajectories
    """
    if filepath.endswith(".gz"):
        import gzip
        with gzip.open(filepath, 'rt') as f:
            return json.load(f)
    else:
        with open(filepath, 'r') as f:
            return json.load(f)


def create_experiment_dir(
    base_dir: str = "./experiments",
    experiment_name: Optional[str] = None
) -> str:
    """Create a new experiment directory.

    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name

    Returns:
        Path to created directory
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"exp_{timestamp}"

    exp_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)

    return exp_dir
