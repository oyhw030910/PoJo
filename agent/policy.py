"""Policy Network Module.

Implements the policy network for RL, combining LLM with actor-critic heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

from .llm_wrapper import LLMWrapper, LLMConfig, LLMOutput


@dataclass
class PolicyConfig:
    """Configuration for policy network.

    Attributes:
        llm_config: LLM configuration
        entropy_coeff: Entropy coefficient
        value_coeff: Value loss coefficient
        clip_param: PPO clip parameter
        max_grad_norm: Maximum gradient norm
        use_value_head: Whether to use value head
        freeze_llm: Whether to freeze LLM weights
        action_space_size: Size of action space (for discrete actions)
    """
    llm_config: Optional[LLMConfig] = None
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    clip_param: float = 0.2
    max_grad_norm: float = 0.5
    use_value_head: bool = True
    freeze_llm: bool = False
    action_space_size: Optional[int] = None


@dataclass
class PolicyOutput:
    """Output from policy forward pass.

    Attributes:
        log_probs: Log probabilities of actions
        values: Value estimates
        entropy: Entropy of policy
        logits: Raw action logits
        hidden_states: Hidden states from LLM
    """
    log_probs: torch.Tensor
    values: torch.Tensor
    entropy: torch.Tensor
    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PolicyNetwork(nn.Module):
    """Policy network for RL with LLM backbone.

    This class implements an actor-critic policy using an LLM as the
    backbone. It supports:
    - Autoregressive action generation
    - Value function estimation
    - Log probability computation
    - Entropy calculation

    Usage:
        config = PolicyConfig(llm_config=LLMConfig())
        policy = PolicyNetwork(config)

        # Get action and log prob
        action, log_prob = policy.get_action_with_log_prob(obs)

        # Forward for training
        outputs = policy.forward_for_training(observations, actions)
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        """Initialize policy network.

        Args:
            config: Policy configuration
        """
        super().__init__()

        self.config = config or PolicyConfig()

        # Initialize LLM backbone
        llm_config = self.config.llm_config or LLMConfig()
        self.llm = LLMWrapper(llm_config)

        # Freeze LLM if specified
        if self.config.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        # Get hidden size
        hidden_size = self.llm.get_hidden_size()

        # Value head
        if self.config.use_value_head:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )
        else:
            self.value_head = None

        # Action embedding for discrete actions
        self.action_space_size = self.config.action_space_size
        if self.action_space_size is not None:
            self.action_embedding = nn.Embedding(
                self.action_space_size,
                hidden_size
            )

        # Device
        self.device = next(self.llm.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        action_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> PolicyOutput:
        """Forward pass through policy.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask
            action_ids: Optional action token IDs for log prob computation
            return_dict: Whether to return dict

        Returns:
            PolicyOutput with log_probs, values, entropy
        """
        batch_size, seq_len = input_ids.shape

        # LLM forward pass
        llm_output = self.llm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=action_ids,  # For log prob computation
        )

        # Get last hidden state
        hidden_states = llm_output.hidden_states
        if hidden_states is None:
            # Fallback: use logits projection
            hidden_states = llm_output.logits

        # Get representation at each position
        # Use last non-padding token for each sequence
        if attention_mask is not None:
            # Get last valid position for each sequence
            last_positions = attention_mask.sum(dim=1) - 1  # [batch]
            batch_indices = torch.arange(batch_size, device=self.device)
            last_hidden = hidden_states[batch_indices, last_positions, :]
        else:
            last_hidden = hidden_states[:, -1, :]

        # Compute value
        if self.value_head is not None:
            values = self.value_head(last_hidden).squeeze(-1)
        else:
            # Use hidden state norm as proxy value
            values = last_hidden.norm(dim=-1)

        # Compute action log probabilities
        if action_ids is not None and llm_output.log_probs is not None:
            # Get log probs for action positions
            action_log_probs = llm_output.log_probs.gather(
                dim=-1,
                index=action_ids.unsqueeze(-1)
            ).squeeze(-1)
            log_probs = action_log_probs.sum(dim=-1)
        else:
            log_probs = torch.zeros(batch_size, device=self.device)

        # Compute entropy
        if llm_output.entropy is not None:
            entropy = llm_output.entropy.mean(dim=-1)
        else:
            entropy = torch.zeros(batch_size, device=self.device)

        # Get logits (distribution over vocabulary/actions)
        logits = llm_output.logits[:, -1, :]  # [batch, vocab]

        return PolicyOutput(
            log_probs=log_probs,
            values=values,
            entropy=entropy,
            logits=logits,
            hidden_states=last_hidden,
        )

    def forward_for_training(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            observations: Observations (converted to input_ids)
            actions: Action token IDs
            attention_mask: Optional attention mask

        Returns:
            Dictionary with:
                - log_probs: [batch, seq_len]
                - values: [batch]
                - entropy: [batch]
        """
        # Handle observation format
        if isinstance(observations, dict):
            input_ids = observations.get("input_ids")
            if attention_mask is None:
                attention_mask = observations.get("attention_mask")
        elif isinstance(observations, torch.Tensor):
            if observations.dtype in [torch.float32, torch.float16]:
                # Already embedded
                input_ids = observations
            else:
                input_ids = observations
        else:
            raise ValueError(f"Unknown observation type: {type(observations)}")

        # Forward pass
        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_ids=actions,
        )

        return {
            "log_probs": output.log_probs,
            "values": output.values,
            "entropy": output.entropy,
            "logits": output.logits,
        }

    @torch.no_grad()
    def get_action(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Get action from policy.

        Args:
            observations: Observations
            attention_mask: Attention mask
            deterministic: Whether to use argmax
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering

        Returns:
            Action token IDs
        """
        # Handle observation format
        if isinstance(observations, dict):
            input_ids = observations.get("input_ids")
            if attention_mask is None:
                attention_mask = observations.get("attention_mask")
        else:
            input_ids = observations

        # Forward pass
        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Get action distribution
        logits = output.logits / temperature

        # Apply top-k filtering
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Apply top-p filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices.argsort(), src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # Sample or take argmax
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return action

    @torch.no_grad()
    def get_action_with_log_prob(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and its log probability.

        Args:
            observations: Observations
            attention_mask: Attention mask
            temperature: Sampling temperature

        Returns:
            Tuple of (action, log_prob)
        """
        # Handle observation format
        if isinstance(observations, dict):
            input_ids = observations.get("input_ids")
            if attention_mask is None:
                attention_mask = observations.get("attention_mask")
        else:
            input_ids = observations

        # Forward pass
        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Get action distribution
        logits = output.logits / temperature
        probs = F.softmax(logits, dim=-1)

        # Sample action
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Compute log probability
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_prob = log_probs.gather(
            dim=-1, index=action.unsqueeze(-1)
        ).squeeze(-1)

        return action, action_log_prob

    @torch.no_grad()
    def get_value(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get value estimate for observations.

        Args:
            observations: Observations
            attention_mask: Attention mask

        Returns:
            Value estimates
        """
        # Handle observation format
        if isinstance(observations, dict):
            input_ids = observations.get("input_ids")
            if attention_mask is None:
                attention_mask = observations.get("attention_mask")
        else:
            input_ids = observations

        # Forward pass
        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return output.values

    def generate_trajectory(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
    ) -> Tuple[str, List[float]]:
        """Generate a trajectory (sequence of actions).

        Args:
            prompt: Initial prompt
            max_length: Maximum generation length
            temperature: Sampling temperature

        Returns:
            Tuple of (generated_text, log_probs)
        """
        self.llm.eval()
        generated_text = ""
        total_log_prob = 0.0

        # Tokenize prompt
        input_ids = self.llm.tokenizer.encode(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)

        for _ in range(max_length):
            # Forward pass
            outputs = self.llm.model(input_ids)
            next_token_logits = outputs.logits[:, -1, :] / temperature

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Compute log prob
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            token_log_prob = log_probs.gather(
                dim=-1, index=next_token
            ).item()
            total_log_prob += token_log_prob

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_text = self.llm.tokenizer.decode(
                input_ids[0],
                skip_special_tokens=True,
            )

            # Check for EOS
            if next_token.item() == self.llm.tokenizer.eos_token_id:
                break

        return generated_text, [total_log_prob]

    def save_pretrained(self, save_dir: str) -> None:
        """Save policy model.

        Args:
            save_dir: Directory to save to
        """
        self.llm.save_pretrained(save_dir)

        # Save value head
        if self.value_head is not None:
            torch.save(
                self.value_head.state_dict(),
                f"{save_dir}/value_head.pt"
            )

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        config: Optional[PolicyConfig] = None
    ) -> "PolicyNetwork":
        """Load from pretrained directory.

        Args:
            model_dir: Directory with model files
            config: Optional config override

        Returns:
            Loaded PolicyNetwork
        """
        if config is None:
            config = PolicyConfig(llm_config=LLMConfig(model_name=model_dir))
        else:
            config.llm_config.model_name = model_dir

        policy = cls(config)

        # Load value head if exists
        value_head_path = f"{model_dir}/value_head.pt"
        if os.path.exists(value_head_path):
            policy.value_head.load_state_dict(
                torch.load(value_head_path)
            )

        return policy


# Import os for from_pretrained
import os
