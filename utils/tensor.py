"""Tensor Utilities Module.

Provides tensor manipulation utilities for RL-LLM Agent.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def to_tensor(
    data: Any,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Convert data to tensor.

    Args:
        data: Data to convert (list, numpy array, dict, etc.)
        device: Target device
        dtype: Target dtype

    Returns:
        Converted tensor
    """
    if isinstance(data, torch.Tensor):
        if device is not None:
            data = data.to(device)
        if dtype is not None:
            data = data.to(dtype)
        return data

    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
        if dtype is not None:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    if isinstance(data, (list, tuple)):
        # Check if nested
        if len(data) > 0 and isinstance(data[0], (list, dict)):
            if isinstance(data[0], dict):
                return {k: to_tensor([d[k] for d in data], device, dtype) for k in data[0].keys()}
            return [to_tensor(d, device, dtype) for d in data]
        tensor = torch.tensor(data, dtype=dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    if isinstance(data, dict):
        return {k: to_tensor(v, device, dtype) for k, v in data.items()}

    # Scalar
    return torch.tensor(data, dtype=dtype or torch.float32, device=device)


def to_numpy(
    tensor: torch.Tensor,
    detach: bool = True,
    cpu: bool = True
) -> np.ndarray:
    """Convert tensor to numpy array.

    Args:
        tensor: Tensor to convert
        detach: Whether to detach from computation graph
        cpu: Whether to move to CPU

    Returns:
        Numpy array
    """
    if detach:
        tensor = tensor.detach()

    if cpu and tensor.is_cuda:
        tensor = tensor.cpu()

    return tensor.numpy()


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False
) -> torch.Tensor:
    """Compute masked mean.

    Args:
        values: Values tensor
        mask: Boolean mask (True = keep)
        dim: Dimension(s) to reduce
        keepdim: Whether to keep reduced dimensions

    Returns:
        Masked mean
    """
    mask = mask.float()
    masked = values * mask

    if dim is None:
        return masked.sum() / (mask.sum() + 1e-8)

    return masked.sum(dim=dim, keepdim=keepdim) / (mask.sum(dim=dim, keepdim=keepdim) + 1e-8)


def masked_var(
    values: torch.Tensor,
    mask: torch.Tensor,
    unbiased: bool = True
) -> torch.Tensor:
    """Compute masked variance.

    Args:
        values: Values tensor
        mask: Boolean mask
        unbiased: Whether to use unbiased estimator

    Returns:
        Masked variance
    """
    mean = masked_mean(values, mask)
    centered_values = values - mean
    var = masked_mean(centered_values ** 2, mask)

    if unbiased:
        mask_sum = mask.sum()
        if mask_sum > 1:
            var = var * mask_sum / (mask_sum - 1)

    return var


def whiten(
    values: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    shift_mean: bool = True,
    unbiased: bool = True
) -> torch.Tensor:
    """Whiten values (normalize to zero mean, unit variance).

    Args:
        values: Values to whiten
        mask: Optional mask
        shift_mean: Whether to shift mean to zero
        unbiased: Whether to use unbiased variance

    Returns:
        Whitened values
    """
    if mask is not None:
        mean = masked_mean(values, mask)
        var = masked_var(values, mask, unbiased=unbiased)
    else:
        mean = values.mean()
        var = values.var(unbiased=unbiased)

    whitened = (values - mean) / torch.sqrt(var + 1e-8)

    if not shift_mean:
        whitened = whitened + mean

    return whitened


def log_probs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """Get log probabilities for labels from logits.

    Args:
        logits: Logits tensor [batch, seq_len, vocab]
        labels: Label tensor [batch, seq_len]

    Returns:
        Log probabilities [batch, seq_len]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def entropy_from_logits(
    logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute entropy from logits.

    Args:
        logits: Logits tensor
        mask: Optional mask

    Returns:
        Entropy values
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)

    if mask is not None:
        entropy = entropy * mask

    return entropy


def kl_divergence(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute KL divergence between two policies.

    Args:
        log_probs: Log probabilities of current policy
        ref_log_probs: Log probabilities of reference policy
        mask: Optional mask

    Returns:
        KL divergence
    """
    kl = ref_log_probs - log_probs

    if mask is not None:
        kl = kl * mask

    return kl


def clip_by_value(
    tensor: torch.Tensor,
    clip_min: float,
    clip_max: float
) -> torch.Tensor:
    """Clip tensor by value.

    Args:
        tensor: Input tensor
        clip_min: Minimum value
        clip_max: Maximum value

    Returns:
        Clipped tensor
    """
    return torch.clamp(tensor, clip_min, clip_max)


def ratio_clipped(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    clip_range: float = 0.2
) -> torch.Tensor:
    """Compute clipped importance sampling ratio.

    Args:
        log_probs: Current log probabilities
        old_log_probs: Old log probabilities
        clip_range: Clipping range

    Returns:
        Clipped ratio
    """
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    return clipped_ratio


def gather_log_probs(
    log_probs: torch.Tensor,
    actions: torch.Tensor
) -> torch.Tensor:
    """Gather log probabilities for specific actions.

    Args:
        log_probs: Log probabilities [batch, seq_len, vocab]
        actions: Action indices [batch, seq_len]

    Returns:
        Gathered log probabilities [batch, seq_len]
    """
    return log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)


def make_attention_mask(
    input_ids: torch.Tensor,
    pad_token_id: int = 0
) -> torch.Tensor:
    """Create attention mask from input IDs.

    Args:
        input_ids: Input token IDs
        pad_token_id: Padding token ID

    Returns:
        Attention mask
    """
    return (input_ids != pad_token_id).long()


def create_causal_mask(
    size: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Create causal (triangular) mask.

    Args:
        size: Mask size
        device: Target device

    Returns:
        Causal mask [size, size]
    """
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int,
    dim: int = 0,
    value: float = 0
) -> torch.Tensor:
    """Pad tensor to multiple of given size.

    Args:
        tensor: Input tensor
        multiple: Multiple to pad to
        dim: Dimension to pad
        value: Padding value

    Returns:
        Padded tensor
    """
    remainder = tensor.size(dim) % multiple
    if remainder == 0:
        return tensor

    padding = multiple - remainder

    # Create padding tuple
    pad_dims = [0] * (2 * tensor.dim())
    pad_dims[2 * (tensor.dim() - dim - 1) + 1] = padding

    return F.pad(tensor, tuple(pad_dims), value=value)


def sequential_merge(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    dim: int = 0
) -> torch.Tensor:
    """Merge two tensors sequentially.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        dim: Dimension to merge along

    Returns:
        Merged tensor
    """
    return torch.cat([tensor1, tensor2], dim=dim)


def batch_select(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    dim: int = 0
) -> torch.Tensor:
    """Select elements from tensor along dimension.

    Args:
        tensor: Input tensor
        indices: Indices to select
        dim: Dimension to select from

    Returns:
        Selected elements
    """
    return tensor.index_select(dim, indices)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    masks: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: Rewards [batch, seq_len]
        values: Value estimates [batch, seq_len]
        next_values: Next state values [batch, seq_len]
        masks: Continuation masks [batch, seq_len]
        gamma: Discount factor
        lam: GAE lambda

    Returns:
        Tuple of (advantages, returns)
    """
    batch_size, seq_len = rewards.shape

    # TD residuals
    deltas = rewards + gamma * next_values * masks - values

    # Initialize
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    # Backward pass
    last_advantage = torch.zeros(batch_size, device=rewards.device)
    last_return = torch.zeros(batch_size, device=rewards.device)

    for t in reversed(range(seq_len)):
        last_advantage = deltas[:, t] + gamma * lam * masks[:, t] * last_advantage
        last_return = rewards[:, t] + gamma * masks[:, t] * last_return

        advantages[:, t] = last_advantage
        returns[:, t] = last_return

    return advantages, returns


def discount_cumsum(
    values: torch.Tensor,
    gamma: float = 0.99,
    masks: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute discounted cumulative sum.

    Args:
        values: Values to discount
        gamma: Discount factor
        masks: Optional continuation masks

    Returns:
        Discounted cumulative sum
    """
    if masks is None:
        masks = torch.ones_like(values)

    result = []
    discounted_sum = torch.zeros(values.shape[0], device=values.device)

    for t in reversed(range(values.shape[1])):
        discounted_sum = values[:, t] + gamma * masks[:, t] * discounted_sum
        result.insert(0, discounted_sum)

    return torch.stack(result, dim=1)
