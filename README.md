# dl-reproducibility-pack

A deep learning reproducibility toolkit for PyTorch and TensorFlow researchers.

## Overview

This skill helps researchers make their deep learning projects fully reproducible for paper publication. It provides:

- Environment templates (requirements.txt, Dockerfile, environment.yml)
- Deterministic training code for PyTorch and TensorFlow
- Standard project structure for research projects
- Configuration management system
- Publication-ready documentation templates

## Installation

### Option 1: Install via Claude Code

```
/plugin marketplace add <this-skill>
```

Or tell Claude Code:

```
我需要你装一个skill，地址是 <owner>/dl-reproducibility-pack，你帮我安装上。
```

### Option 2: Manual Installation

Clone or copy this skill to your Claude Code skills directory:

```bash
# Global installation
~/.claude/skills/dl-reproducibility-pack/

# Or project-local
your-project/.claude/skills/dl-reproducibility-pack/
```

## Usage

### Activating the Skill

Once installed, Claude Code will automatically activate this skill when you:

- Work on PyTorch or TensorFlow research projects
- Ask about reproducibility, random seeds, or determinism
- Need to create environment files or documentation

### Manual Activation

Tell Claude Code:

```
Please use the dl-reproducibility-pack skill to make my project reproducible.
```

## Components

### SKILL.md

The main skill definition containing all templates and best practices for:
- Environment setup (requirements.txt, Dockerfile, environment.yml)
- Random seed setting for PyTorch and TensorFlow
- Project structure
- Configuration management
- README and CITATION.cff templates

### scripts/

Supporting Python utilities (inspired by d2l-zh patterns):

- `reproducibility.py` — Core seed-setting, device detection, Accumulator, EMA, CosineWarmupScheduler, enhanced training/eval loops
- `seed_worker.py` — DataLoader worker seeding, reproducible loader factory, batch samplers
- `config.py` — Type-safe YAML configuration with dataclass-based section management

## Features

### 1. Environment & Dependencies

- requirements.txt with pinned versions
- environment.yml for conda
- Dockerfile for complete containerization

### 2. Random Seeds & Determinism

- PyTorch: `set_seed(seed, framework="pytorch", deterministic=True)`
- TensorFlow: `set_seed(seed, framework="tensorflow")`
- DataLoader worker seeding
- ReproducibilityContext for scoped determinism

### 3. Training Infrastructure (NEW — from d2l patterns)

- **Accumulator** — Clean metric tracking across batches
- **EMA** — Exponential Moving Average for better generalization
- **CosineWarmupScheduler** — LR schedule with linear warmup + cosine decay
- **create_optimizer()** — Factory for AdamW/Adam/SGD/RMSprop
- **grad_clip()** — Gradient clipping for training stability
- **Gradient accumulation** — Simulate large batches on limited GPU memory

### 4. Advanced PyTorch Patterns

- Weight initialization catalog (Kaiming, Xavier, Normal)
- Multi-GPU training (DataParallel + DDP)
- Mixed precision training (AMP)
- Gradient checkpointing
- Data augmentation best practices
- Numerical stability patterns

### 5. Configuration Management

YAML-based hyperparameter management with type-safe Python config classes.

## Example Usage

### Make a Project Reproducible

```bash
# Tell Claude:
"我需要你用 dl-reproducibility-pack 帮我把项目改成可复现的"
```

The skill will analyze your codebase and inject:
1. Deterministic seed setting
2. Weight initialization matching your activations
3. Gradient clipping for stability
4. Cosine warmup LR scheduling
5. EMA (Exponential Moving Average) for better validation
6. Proper project structure and environment files
7. Publication-ready README and CITATION.cff

### Use Seed Setting

```python
from src.reproducibility import set_seed

# PyTorch
set_seed(42, framework="pytorch", deterministic=True)

# TensorFlow
set_seed(42, framework="tensorflow")
```

### Use Reproducible DataLoader

```python
from src.seed_worker import create_reproducible_dataloader

train_loader = create_reproducible_dataloader(
    train_dataset,
    batch_size=128,
    seed=42,
    num_workers=4
)
```

### Use Training Infrastructure

```python
from src.reproducibility import (
    set_seed, get_device, Accumulator,
    EMA, CosineWarmupScheduler, create_optimizer,
    train_one_epoch, evaluate,
)

# Setup
device = get_device()
set_seed(42)

# Optimizer with LR schedule
optimizer = create_optimizer(model, "AdamW", lr=0.001)
scheduler = CosineWarmupScheduler(
    optimizer, warmup_steps=500, total_steps=10000, base_lr=0.001, min_lr=1e-6,
)

# EMA for better validation
ema = EMA(model, decay=0.999)

# Training with gradient accumulation and mixed precision
scaler = torch.amp.GradScaler("cuda")
for epoch in range(num_epochs):
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device,
        scaler=scaler, max_grad_norm=1.0, accumulation_steps=4, ema=ema,
    )
    scheduler.step()
    # Eval with EMA weights
    ema.apply_shadow()
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    ema.restore()
```

### Use Configuration

```python
from src.config import load_config

config = load_config("configs/default.yaml")
print(config.training.batch_size)  # 128
print(config.training.accumulation_steps)  # 1
```

## License

MIT License

## Contributing

Contributions welcome! Please submit issues and pull requests on GitHub.
