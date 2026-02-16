"""
Training loop for the Gothic Dandy Language Model.
Optimized for RTX 4080 with bfloat16, torch.compile(), and Flash Attention.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import ModelConfig, TrainConfig
from src.model import GPT
from src.dataset import create_dataloaders
from src.train_tokenizer import load_tokenizer


def get_lr(step: int, config: TrainConfig) -> float:
    """
    Compute learning rate with linear warmup and cosine decay.
    
    Schedule:
        1. Linear warmup from 0 to max_lr over warmup_iters
        2. Cosine decay from max_lr to min_lr over remaining steps
    """
    if step < config.warmup_iters:
        # Linear warmup
        return config.learning_rate * step / config.warmup_iters
    
    if step > config.lr_decay_iters:
        # Training complete, return min_lr
        return config.min_lr
    
    # Cosine decay
    decay_ratio = (step - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    num_batches: int = 100,
) -> float:
    """Estimate validation loss on a subset of validation data."""
    model.eval()
    losses = []
    
    for i, (x, y) in enumerate(val_loader):
        if i >= num_batches:
            break
        
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    
    model.train()
    return np.mean(losses)


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    path: Path,
) -> None:
    """Save model checkpoint including optimizer state."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "config": model.config,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: Path,
    model: GPT,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda",
) -> Tuple[int, float]:
    """Load checkpoint and restore model/optimizer state."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    step = checkpoint.get("step", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    
    print(f"Checkpoint loaded: {path} (step {step})")
    return step, best_val_loss


def main() -> None:
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train Gothic Dandy Language Model")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--train-file", type=str, default="mixed_train.txt")
    parser.add_argument("--val-file", type=str, default="mixed_val.txt")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--max-iters", type=int, default=None, help="Training iterations (default: 10000, or loaded from checkpoint if resuming)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    
    args = parser.parse_args()
    
    # Setup configuration with user-controlled max_iters
    config = TrainConfig(
        data_dir=args.data_dir,
        train_file=args.train_file,
        val_file=args.val_file,
        tokenizer_path=args.tokenizer,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        compile=args.compile,
    )
    
    # Override max_iters if provided via CLI
    if args.max_iters is not None:
        config.max_iters = args.max_iters
        config.lr_decay_iters = args.max_iters
    
    # Create output directory
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Device setup
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # Check bfloat16 support (RTX 4080 has it)
    if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        print("Warning: bfloat16 not supported, falling back to float16")
        config.dtype = "float16"
    
    # Load tokenizer
    print(f"Loading tokenizer from {config.tokenizer_path}")
    tokenizer = load_tokenizer(config.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # Ensure model_config is initialized
    assert config.model_config is not None, "model_config must be initialized"
    
    # Update model config with actual vocab size
    config.model_config.vocab_size = vocab_size
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_path = Path(config.data_dir) / config.train_file
    val_path = Path(config.data_dir) / config.val_file
    
    train_loader, val_loader = create_dataloaders(
        train_path=str(train_path),
        val_path=str(val_path),
        tokenizer=tokenizer,
        block_size=config.model_config.block_size,
        batch_size=config.batch_size,
        mode="memmap",
    )
    
    # Create model
    print("Initializing model...")
    model = GPT(config.model_config)
    model = model.to(device)
    
    # Compile model for optimization (RTX 4080 benefits significantly)
    if config.compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    
    # Setup automatic mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == "float16"))
    ctx = torch.amp.autocast(device_type="cuda", dtype=getattr(torch, config.dtype))
    
    # TensorBoard logging
    writer = SummaryWriter(log_dir=out_dir / "logs")
    
    # Resume from checkpoint if requested
    start_step = 0
    best_val_loss = float("inf")
    
    if config.resume:
        latest_checkpoint = out_dir / "latest.pt"
        if latest_checkpoint.exists():
            start_step, best_val_loss = load_checkpoint(
                latest_checkpoint, model, optimizer, device
            )
        else:
            print("No checkpoint found to resume from")
    
    # Training loop
    print(f"\nStarting training for {config.max_iters} iterations...")
    print(f"Effective batch size: {config.effective_batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print("=" * 60)
    
    model.train()
    train_iter = iter(train_loader)
    losses = []
    
    pbar = tqdm(range(start_step, config.max_iters), initial=start_step, total=config.max_iters)
    
    for step in pbar:
        # Determine learning rate for this step
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # Gradient accumulation loop
        optimizer.zero_grad()
        loss_accum = 0.0
        
        for micro_step in range(config.gradient_accumulation_steps):
            # Get batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            x, y = x.to(device), y.to(device)
            
            # Forward pass with automatic mixed precision
            with ctx:
                _, loss = model(x, y)
                loss = loss / config.gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            loss_accum += loss.detach().item()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        losses.append(loss_accum)
        
        if step % config.log_interval == 0:
            avg_loss = np.mean(losses[-config.log_interval:])
            perplexity = np.exp(avg_loss)
            pbar.set_description(
                f"step {step} | loss: {avg_loss:.4f} | ppl: {perplexity:.2f} | lr: {lr:.2e}"
            )
            
            writer.add_scalar("train/loss", avg_loss, step)
            writer.add_scalar("train/perplexity", perplexity, step)
            writer.add_scalar("train/learning_rate", lr, step)
        
        # Evaluation
        if step > 0 and step % config.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, device, config.eval_iters)
            val_perplexity = np.exp(val_loss)
            print(f"\n[eval] step {step} | val_loss: {val_loss:.4f} | val_ppl: {val_perplexity:.2f}")
            
            writer.add_scalar("val/loss", val_loss, step)
            writer.add_scalar("val/perplexity", val_perplexity, step)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, step, best_val_loss, out_dir / "best.pt"
                )
            
            model.train()
        
        # Periodic checkpoint
        if step > 0 and step % config.checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, step, best_val_loss, out_dir / "latest.pt"
            )
    
    # Final save
    save_checkpoint(model, optimizer, config.max_iters, best_val_loss, out_dir / "final.pt")
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {out_dir}")
    
    writer.close()


if __name__ == "__main__":
    main()
