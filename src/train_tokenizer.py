"""
Train a Byte-Pair Encoding (BPE) tokenizer on the Gothic Dandy corpus.
Uses HuggingFace tokenizers for efficient subword tokenization.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC


SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>"]


def train_tokenizer(
    files: List[str],
    vocab_size: int = 4096,
    output_path: str = "data/tokenizer.json",
    min_frequency: int = 2,
) -> Tokenizer:
    """
    Train a BPE tokenizer on the provided text files.
    
    Uses byte-level pre-tokenization for universal Unicode handling.
    Special tokens mark document boundaries and padding.
    
    Args:
        files: List of text file paths to train on
        vocab_size: Target vocabulary size (4096-8192 recommended)
        output_path: Where to save the trained tokenizer
        min_frequency: Minimum token frequency to include
        
    Returns:
        Trained Tokenizer instance
    """
    print(f"Training BPE tokenizer with vocab_size={vocab_size}")
    print(f"Training on {len(files)} file(s)")
    
    # Initialize BPE model with byte-level handling
    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    
    # Normalize Unicode to NFKC for consistency
    tokenizer.normalizer = NFKC()
    
    # Byte-level pre-tokenizer handles all Unicode via bytes
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    # Configure trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    
    # Train
    tokenizer.train(files, trainer)
    
    # Enable padding
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<|pad|>"), pad_token="<|pad|>")
    
    # Save
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path_obj))
    
    # Print statistics
    print(f"\nTokenizer saved to: {output_path_obj}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens: {SPECIAL_TOKENS}")
    
    # Test encoding
    test_text = "In the shadow of R'lyeh, the decadent aesthete pondered."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"\nTest encoding:")
    print(f"  Input:  '{test_text}'")
    print(f"  Tokens: {len(encoded.ids)} -> {encoded.ids[:10]}...")
    print(f"  Decoded: '{decoded}'")
    
    return tokenizer


def load_tokenizer(path: str = "data/tokenizer.json") -> Tokenizer:
    """Load a pre-trained tokenizer from disk."""
    tokenizer = Tokenizer.from_file(path)
    return tokenizer


def main() -> None:
    """CLI entry point for tokenizer training."""
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer on Gothic Dandy corpus"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "data/processed/mixed_train.txt",
            "data/processed/wilde_train.txt",
            "data/processed/lovecraft_train.txt",
        ],
        help="Text files to train on",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
        help="Vocabulary size (default: 4096)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tokenizer.json",
        help="Output path (default: data/tokenizer.json)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency (default: 2)",
    )
    
    args = parser.parse_args()
    
    # Filter to existing files
    existing_files = [f for f in args.files if Path(f).exists()]
    if not existing_files:
        print("Error: No valid training files found!")
        print(f"Searched: {args.files}")
        return
    
    train_tokenizer(
        files=existing_files,
        vocab_size=args.vocab_size,
        output_path=args.output,
        min_frequency=args.min_frequency,
    )


if __name__ == "__main__":
    main()
