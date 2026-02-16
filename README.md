# SLM Training Toolkit

A modular training pipeline for Small Language Models (SLM) optimized for consumer GPUs. Includes tools for tokenizer training, model training, and interactive inference.

**Example use case**: Train a ~35M parameter model on custom text corpora (e.g., literary works) to generate stylistically consistent text.

## Overview

This toolkit provides a complete pipeline for training decoder-only transformer models:

1. **Tokenizer Training**: BPE tokenization on your corpus
2. **Model Training**: GPT-style architecture with modern optimizations
3. **Inference**: Both programmatic (CLI) and interactive (TUI) interfaces

**Hardware Requirements**: CUDA-capable GPU recommended (tested on RTX 4080 16GB)

## Features

- **Efficient Architecture**: ~35M parameters, 8 layers, 512 dimensions
- **Modern Optimizations**: bfloat16 mixed precision, torch.compile(), Flash Attention
- **Flexible Training**: Gradient accumulation, cosine learning rate schedule, checkpoint resuming
- **Multiple Interfaces**: CLI for scripting, TUI for exploration
- **Modular Design**: Easy to extend and customize

## Quick Start

```bash
# Install dependencies
uv sync
source .venv/bin/activate

# 1. Train BPE tokenizer on your corpus
gothic-tokenize \
    --files data/processed/mixed_train.txt \
    --vocab-size 4096 \
    --output data/tokenizer.json

# 2. Train model (3-5 hours on RTX 4080)
gothic-train \
    --train-file mixed_train.txt \
    --val-file mixed_val.txt \
    --max-iters 6000 \
    --batch-size 16 \
    --grad-accum 4

# 3. Generate text via CLI
gothic-generate \
    --checkpoint checkpoints/best.pt \
    --prompt "Once upon a time" \
    --max-tokens 200

# 4. Or use the interactive TUI
gothic-tui
```

Monitor training progress: `tensorboard --logdir checkpoints/logs`

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended)
- [uv](https://docs.astral.sh/uv/) package manager

### Install with uv (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd personal-gpt

# Create environment and install dependencies
uv sync

# Activate environment
source .venv/bin/activate

# Verify installation
gothic-tokenize --help
gothic-train --help
```

### Alternative: Install with pip

```bash
python -m venv venv
source venv/bin/activate
pip install -e "."
```

## Project Structure

```
personal-gpt/
├── data/
│   ├── raw/               # Source text files
│   ├── processed/         # Train/validation splits
│   └── tokenizer.json     # Trained BPE tokenizer
├── src/
│   ├── config.py          # Configuration dataclasses
│   ├── model.py           # GPT model implementation
│   ├── train.py           # Training loop
│   ├── generate.py        # CLI generation
│   └── train_tokenizer.py # Tokenizer training
├── chat_tui/
│   └── chat.py            # Interactive TUI interface
├── checkpoints/           # Saved model checkpoints
├── logs/                  # Chat exports and logs
└── pyproject.toml         # Package configuration
```

## Usage Guide

### Training Pipeline

#### Step 1: Prepare Data

Ensure your text data is in `data/processed/` with one document per line:

```bash
# Example structure
data/processed/
├── train.txt     # Training data (90%)
└── val.txt       # Validation data (10%)
```

#### Step 2: Train Tokenizer

```bash
gothic-tokenize \
    --files data/processed/train.txt \
    --vocab-size 4096 \
    --output data/tokenizer.json
```

**Options**:
- `--vocab-size`: 2048 (fast) to 8192 (better coverage)
- `--min-frequency`: Minimum token occurrences (default: 2)

#### Step 3: Train Model

```bash
gothic-train \
    --data-dir data/processed \
    --train-file train.txt \
    --val-file val.txt \
    --tokenizer data/tokenizer.json \
    --out-dir checkpoints \
    --max-iters 6000 \
    --batch-size 16 \
    --grad-accum 4 \
    --lr 3e-4
```

**Key Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-iters` | 6000 | Total training iterations |
| `--batch-size` | 16 | Samples per batch |
| `--grad-accum` | 4 | Gradient accumulation steps |
| `--lr` | 3e-4 | Peak learning rate |
| `--resume` | - | Resume from checkpoint |

**Expected Training Time**: ~3-5 hours (RTX 4080)

**Expected Results**:
- Validation Loss: 2.0-2.5
- Perplexity: 7-12

#### Step 4: Resume Training (if needed)

```bash
gothic-train \
    --resume \
    --out-dir checkpoints \
    --max-iters 10000
```

### Inference

#### CLI Generation

**Single prompt**:
```bash
gothic-generate \
    --checkpoint checkpoints/best.pt \
    --prompt "The story begins" \
    --max-tokens 200 \
    --temperature 0.8 \
    --top-k 40
```

**Interactive mode**:
```bash
gothic-generate \
    --checkpoint checkpoints/best.pt \
    --interactive \
    --temperature 0.9
```

**Sampling Strategy Guide**:

| Mode | Temperature | Top-K | Use Case |
|------|-------------|-------|----------|
| Deterministic | 0.1 | 1 | Reproducible outputs |
| Balanced | 0.8 | 40 | General use |
| Creative | 1.2 | 100 | Exploration/diversity |

#### Interactive TUI

Launch the terminal UI for conversational interaction:

```bash
gothic-tui
```

**Features**:
- Real-time streaming generation
- Model checkpoint switching
- Adjustable generation parameters
- Chat history export

**Slash Commands** (type in TUI):

| Command | Description | Example |
|---------|-------------|---------|
| `/temp <float>` | Set temperature | `/temp 0.9` |
| `/topk <int>` | Set top-k | `/topk 50` |
| `/topp <float>` | Set top-p | `/topp 0.95` |
| `/max <int>` | Set max tokens | `/max 300` |
| `/model <name>` | Switch checkpoint | `/model best` |
| `/clear` | Clear history | `/clear` |
| `/save` | Export chat | `/save` |
| `/help` | Show commands | `/help` |

**Controls**:
- `Enter`: Send message
- `Tab`: Navigate between fields
- `Up/Down`: Scroll history
- `Ctrl+C`: Exit

### Monitoring Training

Launch TensorBoard to view metrics:

```bash
tensorboard --logdir checkpoints/logs
```

View at `http://localhost:6006`:
- Training/validation loss curves
- Perplexity
- Learning rate schedule

## Configuration

### Modifying Hyperparameters

Edit `src/config.py`:

```python
@dataclass
class ModelConfig:
    vocab_size: int = 4096      # BPE vocabulary size
    block_size: int = 384       # Context window length
    n_layer: int = 8            # Transformer layers
    n_head: int = 8             # Attention heads
    n_embd: int = 512           # Embedding dimension

@dataclass
class TrainConfig:
    max_iters: int = 6000
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    warmup_iters: int = 500
```

### Learning Rate Guidelines

| Rate | Use Case |
|------|----------|
| 5e-4 | Fast convergence (risk of instability) |
| 3e-4 | **Recommended** (balanced) |
| 1e-4 | Conservative (large datasets) |
| 1e-5 | Fine-tuning from checkpoint |

### Architecture Scaling

| Component | Default | Reduce If | Increase If |
|-----------|---------|-----------|-------------|
| `block_size` | 384 | OOM errors | Longer context needed |
| `n_layer` | 8 | Slow training | Better quality required |
| `n_embd` | 512 | Memory limited | Richer representations |
| `vocab_size` | 4096 | Fast iteration | Rare word coverage |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA Out of Memory** | Reduce `--batch-size` or increase `--grad-accum`. Disable with `--no-compile`. |
| **Slow Training** | Verify `torch.cuda.is_available()`. Check GPU utilization with `nvidia-smi`. Ensure `--compile` is enabled. |
| **Import Errors** | Run `uv sync` and `source .venv/bin/activate` |
| **Tokenizer Not Found** | Run `gothic-tokenize` before training |
| **Poor Generation Quality** | Increase training iterations. Check validation loss is < 3.0. Adjust temperature. |

## CLI Reference

All commands are defined in `pyproject.toml`:

```toml
[project.scripts]
gothic-tokenize = "src.train_tokenizer:main"
gothic-train = "src.train:main"
gothic-generate = "src.generate:main"
gothic-tui = "chat_tui.chat:main"
```

### gothic-tokenize

Train BPE tokenizer on text corpus.

```bash
gothic-tokenize [OPTIONS]
  --files TEXT          Input text files
  --vocab-size INT      Vocabulary size [default: 4096]
  --output TEXT         Output path [default: data/tokenizer.json]
  --min-frequency INT   Minimum token frequency [default: 2]
```

### gothic-train

Train language model.

```bash
gothic-train [OPTIONS]
  --data-dir TEXT       Data directory [default: data/processed]
  --train-file TEXT     Training file [default: mixed_train.txt]
  --val-file TEXT       Validation file [default: mixed_val.txt]
  --tokenizer TEXT      Tokenizer path [default: data/tokenizer.json]
  --out-dir TEXT        Output directory [default: checkpoints]
  --max-iters INT       Training iterations [default: 6000]
  --batch-size INT      Batch size [default: 16]
  --grad-accum INT      Gradient accumulation [default: 4]
  --lr FLOAT            Learning rate [default: 3e-4]
  --resume              Resume from latest checkpoint
  --compile             Use torch.compile [default: True]
  --no-compile          Disable torch.compile
```

### gothic-generate

Generate text from trained model.

```bash
gothic-generate [OPTIONS]
  --checkpoint TEXT     Model checkpoint (required)
  --tokenizer TEXT      Tokenizer path [default: data/tokenizer.json]
  --prompt TEXT         Generation prompt
  --max-tokens INT      Max tokens to generate [default: 200]
  --temperature FLOAT   Sampling temperature [default: 0.8]
  --top-k INT           Top-k sampling [default: 40]
  --top-p FLOAT         Top-p sampling [default: 0.9]
  --num-samples INT     Number of samples [default: 1]
  --interactive         Interactive mode
```

### gothic-tui

Launch interactive terminal UI.

```bash
gothic-tui
```

## Technical Details

### Model Architecture

- **Type**: Decoder-only transformer (GPT-style)
- **Parameters**: ~35M (8 layers, 8 heads, 512 dim, 4096 vocab)
- **Context**: 384 tokens
- **Features**:
  - Pre-layer normalization
  - Causal self-attention with Flash Attention
  - Weight tying (input/output embeddings)
  - GELU activation, 4x MLP expansion

### Training Configuration

- **Optimizer**: AdamW (β1=0.9, β2=0.95, weight_decay=0.1)
- **Schedule**: Linear warmup (500 steps) + cosine decay
- **Precision**: bfloat16 mixed precision
- **Batching**: Physical batch 16 × 4 accumulation = effective 64
- **Checkpointing**: Best model saved by validation loss

### Performance (RTX 4080)

- **Training Speed**: ~1000-1500 iterations/hour
- **Memory Usage**: ~4-6GB VRAM
- **Optimizations**: torch.compile() provides 20-30% speedup

## Development

### Code Quality

```bash
# Format code
uv run black src/ chat_tui/

# Lint
uv run ruff src/ chat_tui/

# Run tests (if available)
uv run pytest
```

### Project Commands

```bash
# Install dependencies
uv sync

# Run with uv (no activation needed)
uv run gothic-train --max-iters 1000

# Development install
pip install -e "."
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy for transformer implementation
- [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/) for BPE training
- [Project Gutenberg](https://www.gutenberg.org/) for public domain texts
