"""
Configuration dataclasses for the Gothic Dandy Language Model.
Centralizes all hyperparameters for model architecture and training.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """
    GPT-style decoder-only transformer configuration.
    
    Defaults optimized for ~35M parameters on RTX 4080 (16GB).
    """
    vocab_size: int = 4096
    block_size: int = 384
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )
        assert self.block_size > 0, "block_size must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"


@dataclass  
class TrainConfig:
    """
    Training hyperparameters optimized for RTX 4080.
    
    Uses bfloat16 for Tensor Core efficiency on 40-series GPUs.
    Effective batch size = batch_size * gradient_accumulation_steps = 64
    """
    # Data
    data_dir: str = "data/processed"
    train_file: str = "mixed_train.txt"
    val_file: str = "mixed_val.txt"
    tokenizer_path: str = "data/tokenizer.json"
    
    # Model
    model_config: Optional[ModelConfig] = None
    
    # Training loop
    max_iters: int = 6000
    warmup_iters: int = 500
    lr_decay_iters: int = 6000
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 10
    
    # Optimization
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Hardware
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    
    # Checkpointing
    out_dir: str = "checkpoints"
    checkpoint_interval: int = 1000
    resume: bool = False
    resume_checkpoint: Optional[str] = None
    
    # Reproducibility
    seed: int = 1337
    
    def __post_init__(self) -> None:
        """Set defaults and validate configuration."""
        if self.model_config is None:
            self.model_config = ModelConfig()
        if self.lr_decay_iters is None:
            self.lr_decay_iters = self.max_iters
        
        # Validate
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        assert self.warmup_iters < self.max_iters, "warmup_iters must be less than max_iters"
        assert self.dtype in ["float16", "bfloat16", "float32"], (
            f"Invalid dtype: {self.dtype}"
        )
    
    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size accounting for gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps
