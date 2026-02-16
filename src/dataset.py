"""
PyTorch Dataset for efficient text loading and tokenization.
Supports both on-the-fly and pre-tokenized memmap modes.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizers import Tokenizer


class TextDataset(Dataset):
    """
    Dataset for language modeling on tokenized text.
    
    Loads text files, tokenizes them, and creates (input, target) pairs
    where targets are inputs shifted by one position (causal LM).
    
    Supports two modes:
        - "online": Tokenize on-the-fly (simpler, slower)
        - "memmap": Pre-tokenize to uint16 memmap files (faster, recommended)
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Tokenizer,
        block_size: int,
        mode: str = "online",
    ) -> None:
        """
        Initialize dataset.
        
        Args:
            data_path: Path to text file (one document per line)
            tokenizer: Trained tokenizer instance
            block_size: Context window size (max sequence length)
            mode: "online" or "memmap" loading mode
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mode = mode
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        if mode == "online":
            self._init_online_mode()
        elif mode == "memmap":
            self._init_memmap_mode()
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'online' or 'memmap'")
    
    def _init_online_mode(self) -> None:
        """Initialize online tokenization mode."""
        # Load all documents
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.documents = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.documents)} documents from {self.data_path}")
        
        # Tokenize all documents to calculate total tokens
        self.tokenized_docs = []
        total_tokens = 0
        for doc in self.documents:
            encoded = self.tokenizer.encode(doc)
            tokens = encoded.ids
            self.tokenized_docs.append(tokens)
            total_tokens += len(tokens)
        
        # Number of training examples = total tokens // block_size
        self.num_examples = max(1, total_tokens // self.block_size)
        print(f"Total tokens: {total_tokens:,} -> {self.num_examples:,} examples")
    
    def _init_memmap_mode(self) -> None:
        """Initialize memmap mode with pre-tokenized data."""
        # Create memmap path
        memmap_path = self.data_path.with_suffix(".bin")
        
        if not memmap_path.exists():
            print(f"Creating memmap file: {memmap_path}")
            self._create_memmap(memmap_path)
        
        # Load memmap
        self.memmap = np.memmap(memmap_path, dtype=np.uint16, mode="r")
        self.num_examples = (len(self.memmap) - 1) // self.block_size
        print(f"Loaded memmap: {len(self.memmap):,} tokens -> {self.num_examples:,} examples")
    
    def _create_memmap(self, memmap_path: Path) -> None:
        """Pre-tokenize and save to memmap file."""
        # Load and tokenize all text
        with open(self.data_path, "r", encoding="utf-8") as f:
            documents = [line.strip() for line in f if line.strip()]
        
        # Tokenize all documents
        all_tokens = []
        for doc in documents:
            encoded = self.tokenizer.encode(doc)
            all_tokens.extend(encoded.ids)
            # Add EOS token between documents
            eos_id = self.tokenizer.token_to_id("<|endoftext|>")
            all_tokens.append(eos_id)
        
        # Convert to numpy array
        tokens_array = np.array(all_tokens, dtype=np.uint16)
        
        # Save to memmap
        memmap = np.memmap(memmap_path, dtype=np.uint16, mode="w+", shape=tokens_array.shape)
        memmap[:] = tokens_array[:]
        memmap.flush()
        
        print(f"Saved {len(tokens_array):,} tokens to memmap")
    
    def __len__(self) -> int:
        return self.num_examples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training example.
        
        Returns:
            x: Input tokens [block_size]
            y: Target tokens [block_size] (shifted by 1)
        """
        if self.mode == "online":
            return self._get_item_online(idx)
        else:
            return self._get_item_memmap(idx)
    
    def _get_item_online(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item in online mode."""
        # Simple strategy: pick a random document and sample a chunk
        doc_idx = idx % len(self.tokenized_docs)
        tokens = self.tokenized_docs[doc_idx]
        
        if len(tokens) <= self.block_size:
            # Pad if too short
            x = tokens[:-1] if len(tokens) > 1 else tokens
            y = tokens[1:] if len(tokens) > 1 else tokens
            # Pad to block_size
            x = x + [0] * (self.block_size - len(x))
            y = y + [0] * (self.block_size - len(y))
        else:
            # Random start position
            start_idx = np.random.randint(0, len(tokens) - self.block_size)
            chunk = tokens[start_idx:start_idx + self.block_size + 1]
            x = chunk[:-1]
            y = chunk[1:]
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
    def _get_item_memmap(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from memmap."""
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size + 1
        
        # Handle boundary
        if end_idx > len(self.memmap):
            start_idx = len(self.memmap) - self.block_size - 1
            end_idx = len(self.memmap)
        
        chunk = self.memmap[start_idx:end_idx]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        
        return x, y


def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: Tokenizer,
    block_size: int,
    batch_size: int,
    mode: str = "memmap",
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_path: Path to training data file
        val_path: Path to validation data file
        tokenizer: Trained tokenizer
        block_size: Context window size
        batch_size: Batch size per GPU
        mode: "online" or "memmap"
        num_workers: Number of dataloader workers (0 for main thread)
        
    Returns:
        train_loader, val_loader
    """
    train_dataset = TextDataset(train_path, tokenizer, block_size, mode)
    val_dataset = TextDataset(val_path, tokenizer, block_size, mode)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
