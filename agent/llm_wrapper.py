"""LLM Wrapper Module.

Provides a unified interface for various LLM backends.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class LLMConfig:
    """Configuration for LLM wrapper.

    Attributes:
        model_name: HuggingFace model name or path
        dtype: Model dtype ('float16', 'float32', 'bfloat16')
        device: Device to load model on
        max_seq_length: Maximum sequence length
        lora_enabled: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_target_modules: Target modules for LoRA
        padding_side: Padding side for tokenizer
    """
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "float16"
    device: str = "auto"  # Auto-detect: cuda > mps > cpu
    max_seq_length: int = 2048
    lora_enabled: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    padding_side: str = "left"


@dataclass
class LLMOutput:
    """Output from LLM forward pass.

    Attributes:
        logits: Raw logits from model
        hidden_states: Hidden states
        attention_mask: Attention mask used
        log_probs: Log probabilities for actions
        entropy: Entropy of distribution
    """
    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    log_probs: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMWrapper(nn.Module):
    """Wrapper for HuggingFace causal language models.

    This class provides a unified interface for loading and using
    various LLM backends with support for:
    - LoRA fine-tuning
    - Custom tokenization
    - Log probability computation
    - Value head attachment

    Usage:
        config = LLMConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct")
        llm = LLMWrapper(config)
        outputs = llm.generate(prompt, max_length=100)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM wrapper.

        Args:
            config: LLM configuration
        """
        super().__init__()

        self.config = config or LLMConfig()

        # Parse dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = dtype_map.get(self.config.dtype, torch.float16)

        # Set device - check for CUDA, MPS, then CPU
        # Auto-detect best available device
        if self.config.device in ["auto", "", None]:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            # Use specified device
            self.device = torch.device(self.config.device)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            padding_side=self.config.padding_side,
            trust_remote_code=True,
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        use_device_map = self.config.device == "cuda" or (
            hasattr(torch.backends, "mps") and
            torch.backends.mps.is_available() and
            self.config.device != "cpu"
        )

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.dtype,
            device_map="auto" if use_device_map else None,
            trust_remote_code=True,
        )

        # Move to device if not using device_map
        if not use_device_map:
            self.model.to(self.device)

        # Apply LoRA if enabled
        if self.config.lora_enabled:
            self._apply_lora()

        # Resize embedding for special tokens if needed
        self._setup_special_tokens()

        # Value head for actor-critic
        self.value_head: Optional[nn.Linear] = None

    def _apply_lora(self) -> None:
        """Apply LoRA fine-tuning configuration."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            modules_to_save=None,
        )
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"LoRA applied: {trainable_params:,} trainable / {total_params:,} total "
              f"({100 * trainable_params / total_params:.2f}%)")

    def _setup_special_tokens(self) -> None:
        """Setup special tokens for RL."""
        # Add special tokens for actions if needed
        special_tokens = {
            "additional_special_tokens": [
                "<action>", "</action>",
                "<observation>", "</observation>",
                "<reward>", "</reward>",
            ]
        }
        num_added = self.tokenizer.add_special_tokens(
            special_tokens,
            replace_additional_special_tokens=False
        )
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def add_value_head(self, hidden_size: Optional[int] = None) -> None:
        """Add a value head for actor-critic methods.

        Args:
            hidden_size: Hidden size (defaults to model hidden size)
        """
        if hidden_size is None:
            hidden_size = self.model.config.hidden_size

        self.value_head = nn.Linear(hidden_size, 1)
        self.value_head.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> LLMOutput:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            labels: Optional labels for loss computation
            return_dict: Whether to return dict or tuple
            **kwargs: Additional arguments

        Returns:
            LLMOutput with logits and other outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            return_dict=True,
            **kwargs,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else None

        # Compute log probabilities
        log_probs = None
        entropy = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask = attention_mask[..., 1:].contiguous() if attention_mask is not None else None

            # Compute log probs
            log_probs = torch.log_softmax(shift_logits, dim=-1)

            # Compute entropy
            probs = torch.softmax(shift_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

            # Apply mask
            if shift_attention_mask is not None:
                log_probs = log_probs * shift_attention_mask.unsqueeze(-1)
                entropy = entropy * shift_attention_mask

        return LLMOutput(
            logits=logits,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            log_probs=log_probs,
            entropy=entropy,
            metadata={"loss": outputs.loss if hasattr(outputs, 'loss') else None},
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[int], torch.Tensor],
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> Union[str, List[str], torch.Tensor]:
        """Generate text from prompt.

        Args:
            prompt: Input prompt (string, token list, or tensor)
            max_length: Maximum total length
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling
            top_k: Top-k sampling
            do_sample: Whether to sample
            num_return_sequences: Number of sequences to return
            pad_token_id: Padding token ID
            eos_token_id: EOS token ID
            **kwargs: Additional arguments

        Returns:
            Generated text or tokens
        """
        # Tokenize prompt if string
        if isinstance(prompt, str):
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
        elif isinstance(prompt, list):
            input_ids = torch.tensor([prompt], device=self.device)
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            input_ids = prompt.to(self.device)
            attention_mask = torch.ones_like(input_ids, device=self.device)

        # Set defaults
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        if max_length is None and max_new_tokens is None:
            max_length = self.config.max_seq_length

        # Generate
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        # Decode if single sequence
        if num_return_sequences == 1 and isinstance(prompt, str):
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Return list of strings
        if isinstance(prompt, str):
            return [
                self.tokenizer.decode(seq, skip_special_tokens=True)
                for seq in outputs
            ]

        return outputs

    @torch.no_grad()
    def get_action_log_probs(
        self,
        input_ids: torch.Tensor,
        action_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get log probabilities for action tokens.

        Args:
            input_ids: Input token IDs (prompt)
            action_ids: Action token IDs to evaluate
            attention_mask: Attention mask

        Returns:
            Log probabilities for actions
        """
        # Concatenate prompt and actions
        full_ids = torch.cat([input_ids, action_ids], dim=-1)

        if attention_mask is not None:
            full_mask = torch.cat([
                attention_mask,
                torch.ones_like(action_ids)
            ], dim=-1)
        else:
            full_mask = torch.ones_like(full_ids)

        # Forward pass
        outputs = self.forward(
            full_ids,
            attention_mask=full_mask,
        )

        # Get log probs for action positions
        logits = outputs.logits[:, input_ids.shape[1]-1:-1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        # Gather log probs for actual action tokens
        action_log_probs = log_probs.gather(
            dim=-1,
            index=action_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Sum over action sequence
        return action_log_probs.sum(dim=-1)

    def get_hidden_size(self) -> int:
        """Get model hidden size.

        Returns:
            Hidden size
        """
        return self.model.config.hidden_size

    def get_num_layers(self) -> int:
        """Get number of transformer layers.

        Returns:
            Number of layers
        """
        return self.model.config.num_hidden_layers

    def get_vocab_size(self) -> int:
        """Get vocabulary size.

        Returns:
            Vocabulary size
        """
        return len(self.tokenizer)

    def save_pretrained(self, save_dir: str) -> None:
        """Save model and tokenizer.

        Args:
            save_dir: Directory to save to
        """
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        config: Optional[LLMConfig] = None
    ) -> "LLMWrapper":
        """Load from pretrained directory.

        Args:
            model_dir: Directory with model files
            config: Optional config override

        Returns:
            Loaded LLMWrapper
        """
        if config is None:
            config = LLMConfig(model_name=model_dir)
        else:
            config.model_name = model_dir

        return cls(config)
