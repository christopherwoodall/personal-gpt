"""
Interactive text generation for the Gothic Dandy Language Model.
CLI tool for sampling from trained checkpoints with various strategies.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import ModelConfig
from src.model import GPT
from src.train_tokenizer import load_tokenizer


def interactive_chat(
    model: GPT,
    tokenizer,
    device: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
) -> None:
    """
    Run interactive chat session with the model.
    
    Continuously prompts for input and generates responses until 'quit'.
    """
    print("\n" + "=" * 60)
    print("GOTHIC DANDY INTERACTIVE CHAT")
    print("=" * 60)
    print(f"Settings: temperature={temperature}, top_k={top_k}, top_p={top_p}")
    print("Type 'quit' to exit, 'settings' to change parameters")
    print("=" * 60 + "\n")
    
    while True:
        # Get user input
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not prompt:
            continue
        
        if prompt.lower() == "quit":
            print("Goodbye!")
            break
        
        if prompt.lower() == "settings":
            print(f"\nCurrent settings:")
            print(f"  max_tokens: {max_tokens}")
            print(f"  temperature: {temperature}")
            print(f"  top_k: {top_k}")
            print(f"  top_p: {top_p}")
            print("\nChange with: --max-tokens, --temperature, --top-k, --top-p")
            continue
        
        # Encode prompt
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
        
        # Generate
        print("\nGenerating...\n")
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
        
        # Decode and print
        generated_ids = output_ids[0].tolist()
        generated_text = tokenizer.decode(generated_ids)
        
        # Only print the new part (after the prompt)
        print(generated_text)
        print("\n" + "-" * 60 + "\n")


def generate_single(
    model: GPT,
    tokenizer,
    device: str,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    num_samples: int = 1,
) -> None:
    """Generate one or more samples from a prompt."""
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    
    print(f"\nPrompt: {prompt}")
    print("=" * 60)
    
    for i in range(num_samples):
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
        
        generated_ids = output_ids[0].tolist()
        generated_text = tokenizer.decode(generated_ids)
        
        if num_samples > 1:
            print(f"\n--- Sample {i + 1} ---")
        print(generated_text)
        print()


def main() -> None:
    """CLI entry point for text generation."""
    parser = argparse.ArgumentParser(
        description="Generate text with Gothic Dandy Language Model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/tokenizer.json",
        help="Path to tokenizer (default: data/tokenizer.json)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="In the shadow of",
        help="Text prompt to complete",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate (default: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling (default: 40, 0=disabled)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling (default: 0.9, 0=disabled)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate (default: 1)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive chat mode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    # Load tokenizer
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer not found: {tokenizer_path}")
        return
    
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = load_tokenizer(str(tokenizer_path))
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint or create default
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        print("Warning: No config in checkpoint, using defaults")
        config = ModelConfig()
        config.vocab_size = tokenizer.get_vocab_size()
    
    # Create and load model
    print("Initializing model...")
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {model.count_parameters():,} parameters")
    print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")
    print(f"Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    # Convert 0 to None for disabled sampling
    top_k = args.top_k if args.top_k > 0 else None
    top_p = args.top_p if args.top_p > 0.0 else None
    
    # Run generation
    if args.interactive:
        interactive_chat(
            model,
            tokenizer,
            args.device,
            args.max_tokens,
            args.temperature,
            top_k if top_k else 40,
            top_p if top_p else 0.9,
        )
    else:
        generate_single(
            model,
            tokenizer,
            args.device,
            args.prompt,
            args.max_tokens,
            args.temperature,
            top_k,
            top_p,
            args.num_samples,
        )


if __name__ == "__main__":
    main()
