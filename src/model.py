"""
GPT-style decoder-only transformer for causal language modeling.
Clean nanoGPT implementation with Flash Attention and weight tying.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with Flash Attention support.
    
    Uses torch.nn.functional.scaled_dot_product_attention for hardware-
    optimized attention computation on RTX 4080.
    """
    
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, query, value projections for all heads (concatenated)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Flash Attention flag (checked at runtime)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        
        if not self.flash:
            print("WARNING: Flash Attention not available, using manual attention")
            # Create causal mask manually
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            
        Returns:
            Output tensor [batch_size, seq_len, n_embd]
        """
        B, T, C = x.size()
        
        # Compute query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention: (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Causal self-attention
        if self.flash:
            # Flash Attention: efficient hardware-kernel fusion
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # Manual attention with causal mask
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Feed-forward network with GELU activation.
    Standard transformer MLP: expand 4x, GELU, project back.
    """
    
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        # Use tanh approximation for faster GELU on GPU
        self.act = nn.GELU(approximate="tanh")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block with pre-normalization.
    
    Architecture: LayerNorm -> Attention -> LayerNorm -> MLP
    Pre-norm provides better gradient flow for deep networks.
    """
    
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: residual connection outside layer norm
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    GPT-style decoder-only transformer for causal language modeling.
    
    Architecture:
        - Token embeddings (weight-tied with output head)
        - Positional embeddings (learned)
        - N transformer blocks
        - Final layer norm
        - Language modeling head
    
    Weight tying reduces parameters by ~vocab_size * n_embd (~2M for our config).
    """
    
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        
        # Language modeling head (weight-tied with wte)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: share embeddings between input and output
        self.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Scale residual projections for training stability
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )
        
        print(f"Model initialized: {self.count_parameters():,} parameters")
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with small normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def count_parameters(self) -> int:
        """Count non-embedding trainable parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        # Subtract non-trainable position embeddings and shared weight
        n_params -= self.wpe.weight.numel()
        return n_params
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            idx: Input token indices [batch_size, seq_len]
            targets: Target token indices [batch_size, seq_len] (for training)
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            loss: Cross-entropy loss (if targets provided)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Sequence length {t} exceeds block size {self.config.block_size}"
        )
        
        # Token embeddings
        tok_emb = self.wte(idx)  # [b, t, n_embd]
        
        # Position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.wpe(pos)  # [1, t, n_embd]
        
        # Combine and dropout
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.h:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy: [b*t, vocab_size] vs [b*t]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            idx: Starting token indices [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = greedy, lower = more focused)
            top_k: Limit sampling to top-k tokens (None = disabled)
            top_p: Nucleus sampling threshold (None = disabled)
            
        Returns:
            Generated token indices including prompt
        """
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # [batch_size, vocab_size]
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            
            # Top-p (nucleus) filtering
            if top_p is not None and top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative prob above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
