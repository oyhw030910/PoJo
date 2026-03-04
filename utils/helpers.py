"""Helper Utilities Module.

Provides general utility functions for RL-LLM Agent.
"""

import random
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import json
import os


def seed_all(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_tokens(
    tokens: Union[List[int], torch.Tensor],
    tokenizer: Optional[Any] = None,
    truncate: int = 100
) -> str:
    """Format tokens for display.

    Args:
        tokens: Token IDs
        tokenizer: Optional tokenizer for decoding
        truncate: Max tokens to display

    Returns:
        Formatted string
    """
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()

    if len(tokens) > truncate:
        display_tokens = tokens[:truncate]
        suffix = f"... ({len(tokens) - truncate} more)"
    else:
        display_tokens = tokens
        suffix = ""

    if tokenizer:
        try:
            text = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(display_tokens)
            )
            return f"{text}{suffix}"
        except Exception:
            pass

    # Fallback: show token IDs
    return f"[{', '.join(map(str, display_tokens))}]{suffix}"


def to_tensor(
    data: Any,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Convert data to tensor.

    Args:
        data: Data to convert
        device: Target device
        dtype: Target dtype

    Returns:
        Converted tensor
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, dtype)

    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor.to(device)

    if isinstance(data, (list, tuple)):
        tensor = torch.tensor(data, dtype=dtype)
        return tensor.to(device)

    if isinstance(data, (int, float)):
        return torch.tensor(data, dtype=dtype or torch.float32, device=device)

    raise ValueError(f"Cannot convert {type(data)} to tensor")


def to_numpy(
    tensor: torch.Tensor,
    detach: bool = True
) -> np.ndarray:
    """Convert tensor to numpy array.

    Args:
        tensor: Tensor to convert
        detach: Whether to detach from graph

    Returns:
        Numpy array
    """
    if detach:
        tensor = tensor.detach()

    if tensor.is_cuda:
        tensor = tensor.cpu()

    return tensor.numpy()


def collate_fn(
    batch: List[Dict[str, Any]],
    device: Union[str, torch.device] = "cpu",
    pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """Collate function for data loaders.

    Args:
        batch: List of samples
        device: Target device
        pad_token_id: Padding token ID

    Returns:
        Collated batch dictionary
    """
    # Get max length
    max_len = max(len(sample.get("input_ids", [])) for sample in batch)

    # Pad and convert
    input_ids = []
    attention_masks = []
    labels = []
    rewards = []
    actions = []

    for sample in batch:
        # Pad input_ids
        ids = sample.get("input_ids", [])
        mask = sample.get("attention_mask", [1] * len(ids))
        label = sample.get("labels", ids.copy())

        # Pad to max length
        padding = max_len - len(ids)
        ids = ids + [pad_token_id] * padding
        mask = mask + [0] * padding
        label = label + [-100] * padding  # Ignore padding in loss

        input_ids.append(ids)
        attention_masks.append(mask)
        labels.append(label)

        if "reward" in sample:
            rewards.append(sample["reward"])
        if "action" in sample:
            actions.append(sample["action"])

    result = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }

    if rewards:
        result["rewards"] = torch.tensor(rewards, dtype=torch.float32, device=device)
    if actions:
        result["actions"] = torch.tensor(actions, dtype=torch.long, device=device)

    return result


def pad_sequences(
    sequences: List[List[int]],
    padding_value: int = 0,
    max_length: Optional[int] = None,
    padding_side: str = "right"
) -> torch.Tensor:
    """Pad sequences to same length.

    Args:
        sequences: List of sequences
        padding_value: Padding value
        max_length: Maximum length (None for longest)
        padding_side: Padding side ('left' or 'right')

    Returns:
        Padded tensor
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        padding = max_length - len(seq)
        if padding_side == "left":
            padded_seq = [padding_value] * padding + seq
        else:
            padded_seq = seq + [padding_value] * padding
        padded.append(padded_seq)

    return torch.tensor(padded, dtype=torch.long)


def truncate_sequences(
    sequences: List[List[int]],
    max_length: int
) -> List[List[int]]:
    """Truncate sequences to max length.

    Args:
        sequences: List of sequences
        max_length: Maximum length

    Returns:
        Truncated sequences
    """
    return [seq[:max_length] for seq in sequences]


def compute_returns(
    rewards: List[float],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    values: Optional[List[float]] = None
) -> List[float]:
    """Compute discounted returns.

    Args:
        rewards: List of rewards
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        values: Optional value estimates for GAE

    Returns:
        List of returns
    """
    if values is not None:
        # Use GAE
        returns = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * gae_lambda * gae
            returns.insert(0, gae + values[t])

        return returns
    else:
        # Simple discounted returns
        returns = []
        discounted_sum = 0

        for reward in reversed(rewards):
            discounted_sum = reward + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        return returns


def normalize_rewards(
    rewards: List[float],
    epsilon: float = 1e-8
) -> List[float]:
    """Normalize rewards.

    Args:
        rewards: List of rewards
        epsilon: Small value for numerical stability

    Returns:
        Normalized rewards
    """
    mean = np.mean(rewards)
    std = np.std(rewards)

    if std < epsilon:
        return [0.0] * len(rewards)

    return [(r - mean) / std for r in rewards]


def explained_variance(
    predicted: List[float],
    actual: List[float]
) -> float:
    """Compute explained variance.

    Args:
        predicted: Predicted values
        actual: Actual values

    Returns:
        Explained variance ratio
    """
    if len(predicted) != len(actual) or len(predicted) == 0:
        return 0.0

    predicted = np.array(predicted)
    actual = np.array(actual)

    var_actual = np.var(actual)
    if var_actual < 1e-8:
        return 0.0

    cov = np.mean((predicted - predicted.mean()) * (actual - actual.mean()))
    return cov / var_actual


def safe_json_serialize(obj: Any) -> Any:
    """Safely serialize object to JSON-compatible format.

    Args:
        obj: Object to serialize

    Returns:
        JSON-compatible representation
    """
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): safe_json_serialize(v) for k, v in obj.items()}
    elif hasattr(obj, 'item'):  # numpy types
        return obj.item()
    elif hasattr(obj, 'tolist'):  # arrays/tensors
        return obj.tolist()
    else:
        return str(obj)


def save_json(
    data: Any,
    filepath: str,
    indent: int = 2
) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(safe_json_serialize(data), f, indent=indent, default=str)


def load_json(filepath: str) -> Any:
    """Load data from JSON file.

    Args:
        filepath: Input file path

    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_device() -> torch.device:
    """Get the best available device.

    Returns:
        Device object
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters.

    Args:
        model: Model to count

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


def format_number(num: int) -> str:
    """Format large numbers with K/M/B suffixes.

    Args:
        num: Number to format

    Returns:
        Formatted string
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def batchify(
    items: List[Any],
    batch_size: int
) -> List[List[Any]]:
    """Split items into batches.

    Args:
        items: Items to batch
        batch_size: Batch size

    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def flatten(
    nested: List[List[Any]]
) -> List[Any]:
    """Flatten nested list.

    Args:
        nested: Nested list

    Returns:
        Flattened list
    """
    return [item for sublist in nested for item in sublist]
