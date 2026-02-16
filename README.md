# Gothic Dandy SLM

A ~35M parameter language model trained on the cosmic horror of H.P. Lovecraft and the decadent wit of Oscar Wilde.

## Features

- **Architecture**: GPT-style decoder-only transformer (~35M params, 8 layers, 512 dim)
- **Tokenization**: BPE with 4096 vocab, trained on mixed Wilde/Lovecraft corpus
- **Optimizations**: bfloat16, torch.compile(), Flash Attention, gradient accumulation
- **Hardware**: Optimized for RTX 4080 (16GB VRAM)

## Quick Start

```bash
# Install with uv
uv sync
source .venv/bin/activate

# 1. Train tokenizer
gothic-tokenize --files data/processed/mixed_train.txt --vocab-size 4096

# 2. Train model (~3-5 hours on RTX 4080)
gothic-train \
    --train-file mixed_train.txt \
    --val-file mixed_val.txt \
    --max-iters 6000 \
    --batch-size 16 \
    --grad-accum 4

# 3. Generate text
gothic-chat --checkpoint checkpoints/best.pt --interactive
```

Monitor training with TensorBoard: `tensorboard --logdir checkpoints/logs`

## Installation

**Prerequisites**: Python 3.10+, CUDA GPU, [uv](https://docs.astral.sh/uv/)

```bash
# Clone and install
git clone <repository-url>
cd personal-gpt
uv sync
source .venv/bin/activate

# Verify
gothic-tokenize --help
```

**Alternative (pip)**:
```bash
python -m venv venv && source venv/bin/activate
pip install -e "."
```

## Project Structure

```
personal-gpt/
├── data/
│   ├── raw/               # Project Gutenberg texts
│   ├── processed/         # Train/val splits
│   └── tokenizer.json     # BPE tokenizer
├── src/
│   ├── config.py          # Hyperparameters
│   ├── model.py           # GPT architecture
│   ├── train.py           # Training loop
│   ├── generate.py        # Text generation
│   └── train_tokenizer.py # BPE training
├── checkpoints/           # Saved models
└── pyproject.toml         # Package config
```

## Usage Guide

### Training from Scratch

```bash
# Step 1: Train tokenizer (2-3 min)
gothic-tokenize --files data/processed/mixed_train.txt --vocab-size 4096

# Step 2: Train model (3-5 hours)
gothic-train \
    --train-file mixed_train.txt \
    --val-file mixed_val.txt \
    --max-iters 6000 \
    --batch-size 16 \
    --grad-accum 4 \
    --lr 3e-4

# Monitor: tensorboard --logdir checkpoints/logs
```

**Expected Results** (after 6000 iters):
- Val Loss: 2.0-2.5
- Perplexity: 7-12
- Training time: ~3-5 hrs (RTX 4080)

### Individual Authors

Train on Wilde only:
```bash
gothic-tokenize --files data/processed/wilde_train.txt --output data/tokenizer_wilde.json
gothic-train --train-file wilde_train.txt --val-file wilde_val.txt \
    --tokenizer data/tokenizer_wilde.json --out-dir checkpoints_wilde
```

Train on Lovecraft only:
```bash
gothic-tokenize --files data/processed/lovecraft_train.txt --output data/tokenizer_lovecraft.json
gothic-train --train-file lovecraft_train.txt --val-file lovecraft_val.txt \
    --tokenizer data/tokenizer_lovecraft.json --out-dir checkpoints_lovecraft
```

### Resuming Training

```bash
gothic-train --resume --out-dir checkpoints --max-iters 6000
# Continues from checkpoints/latest.pt
```

### Generation Modes

**Interactive** (recommended):
```bash
gothic-chat --checkpoint checkpoints/best.pt --interactive --temperature 0.9
```

**Single prompt**:
```bash
gothic-chat \
    --checkpoint checkpoints/best.pt \
    --prompt "In the shadow of" \
    --max-tokens 200 \
    --temperature 0.8 \
    --top-k 40
```

**Sampling strategies**:
| Mode | Temperature | Top-K | Use Case |
|------|-------------|-------|----------|
| Greedy | 0.1 | 1 | Deterministic |
| Balanced | 0.8 | 40 | Default |
| Creative | 1.2 | 100 | Exploration |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **CUDA OOM** | Reduce `--batch-size 8 --grad-accum 8` or use `--no-compile` |
| **Slow training** | Check `torch.cuda.is_available()`, ensure GPU util > 80% |
| **Import errors** | Run `uv sync` and `source .venv/bin/activate` |
| **Tokenizer not found** | Run `gothic-tokenize` first |

## Configuration

### Key Hyperparameters

Edit `src/config.py` or use CLI flags:

```python
@dataclass
class TrainConfig:
    max_iters: int = 6000          # Training iterations
    batch_size: int = 16           # Per-GPU batch
    gradient_accumulation_steps: int = 4  # Effective batch = 64
    learning_rate: float = 3e-4    # Peak LR
    warmup_iters: int = 500        # Warmup steps
```

### Learning Rate Guide

| Rate | Use Case |
|------|----------|
| 5e-4 | Fast convergence (small data) |
| 3e-4 | **Default** (balanced) |
| 1e-4 | Large dataset, stability |
| 1e-5 | Fine-tuning from checkpoint |

### Architecture Tuning

| Parameter | Default | Trade-off |
|-----------|---------|-----------|
| `block_size` | 384 | ↑ = longer context, ↑ memory |
| `vocab_size` | 4096 | ↓ = faster, ↑ = better coverage |
| `n_layer` | 8 | ↓ = faster, ↑ = more capacity |
| `n_embd` | 512 | ↓ = faster, ↑ = richer representations |

Modify in `src/config.py`:
```python
@dataclass
class ModelConfig:
    block_size: int = 512    # Increase for longer context
    n_layer: int = 6         # Reduce if OOM
```

## CLI Reference

**Entry points** (defined in `pyproject.toml`):
```toml
gothic-tokenize = "src.train_tokenizer:main"
gothic-train = "src.train:main"
gothic-chat = "src.generate:main"
```

### gothic-tokenize

```bash
gothic-tokenize [OPTIONS]
  --files TEXT         Training files [default: mixed_train.txt]
  --vocab-size INT     Vocabulary size [default: 4096]
  --output TEXT        Output path [default: data/tokenizer.json]
  --min-frequency INT  Min token frequency [default: 2]
```

### gothic-train

```bash
gothic-train [OPTIONS]
  --data-dir TEXT      Data directory [default: data/processed]
  --train-file TEXT    Training file [default: mixed_train.txt]
  --val-file TEXT      Validation file [default: mixed_val.txt]
  --tokenizer TEXT     Tokenizer path [default: data/tokenizer.json]
  --out-dir TEXT       Output directory [default: checkpoints]
  --resume             Resume from latest checkpoint
  --max-iters INT      Training iterations [default: 6000]
  --batch-size INT     Batch size [default: 16]
  --grad-accum INT     Gradient accumulation [default: 4]
  --lr FLOAT           Learning rate [default: 3e-4]
  --compile            Use torch.compile() [default: True]
  --no-compile         Disable torch.compile()
```

### gothic-chat

```bash
gothic-chat [OPTIONS]
  --checkpoint TEXT    Model checkpoint (required)
  --tokenizer TEXT     Tokenizer path [default: data/tokenizer.json]
  --prompt TEXT        Generation prompt [default: "In the shadow of"]
  --max-tokens INT     Max tokens [default: 200]
  --temperature FLOAT  Sampling temp [default: 0.8]
  --top-k INT          Top-k sampling [default: 40]
  --top-p FLOAT        Top-p sampling [default: 0.9]
  --num-samples INT    Number of samples [default: 1]
  --interactive        Interactive mode
  --device TEXT        Device [default: cuda]
```

## Architecture Details

**GPT Model** (~35M parameters):
- Token + positional embeddings (weight-tied)
- 8 transformer blocks (pre-norm)
- Causal self-attention with Flash Attention
- MLP: 4x expansion, GELU activation

**Training Loop**:
- Mixed precision: bfloat16 forward/backward
- Optimizer: AdamW (lr=3e-4, wd=0.1)
- Schedule: Linear warmup (500 steps) + cosine decay
- Gradient accumulation: effective batch = 64
- Checkpointing: best model saved on validation loss

## Development

```bash
# Setup
uv sync
source .venv/bin/activate

# Run without activating
uv run gothic-train --max-iters 1000

# Code quality
uv run black src/
uv run ruff src/

# Test imports
python -c "from src.config import ModelConfig; print('✓ config')"
python -c "from src.model import GPT; print('✓ model')"
```

## Hardware Specs

**Tested on**: RTX 4080 (16GB VRAM)

**Performance**:
- bfloat16 + torch.compile(): ~20-30% speedup
- Flash Attention: Memory-efficient on Ampere+
- Training: ~1000-1500 iters/hour

**Memory Usage**:
- Model: ~140MB
- Batch (16×384): ~200MB
- Activations: ~2-3GB
- Total: ~4-6GB (fits comfortably in 16GB)

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- **nanoGPT** by Andrej Karpathy for the clean transformer implementation
- **HuggingFace tokenizers** for efficient BPE training
- **Project Gutenberg** for the public domain texts
