# dl-reproducibility-pack v3

A deep learning reproducibility toolkit for PyTorch and TensorFlow researchers.  
**v3** incorporates 2025 AAAI/Nature reproducibility standards and PyTorch 2.6/2.7 features.

---

## What's New in v3 (vs v2)

v3 adds **12 major improvements** based on AAAI 2025 Reproducibility Checklist,
PyTorch 2.6/2.7 release notes, and industry best practices.

| # | Improvement | What it does |
|---|-------------|--------------|
| 1 | **Full RNG Checkpoint** | Save/restore Python, NumPy, PyTorch, and CUDA RNG states for bit-exact resume |
| 2 | **Two-Run Verification** | `verify_reproducibility()` — assert identical loss curves with same seed |
| 3 | **Environment Locking** | `lock_environment()` — pip freeze ALL packages + nvidia-smi + hardware |
| 4 | **Dataset Versioning** | `DatasetVersioning` + `save/load_dataset_split` + `stratified_split` |
| 5 | **FSDP2 Training** | `create_distributed_model(strategy="fsdp2")` — fully_shard with DTensor |
| 6 | **Advanced torch.compile** | `compile_model(use_mega_cache=True)` — PyTorch 2.7 Mega Cache + Prologue Fusion |
| 7 | **Profiling** | `ProfileContext` + `BenchmarkTimer` + `benchmark_model` throughput testing |
| 8 | **TorchMetrics** | `get_metrics()` — standardized Accuracy/F1/AUROC/Precision/Recall/MeanIoU |
| 9 | **Experiment Tracking** | `ExperimentTracker` — unified Trackio/MLflow/W&B interface |
| 10 | **TorchElastic** | `get_elastic_environment()` — fault-tolerant training with auto-recovery |
| 11 | **Multi-Platform** | `get_device()` supports CUDA + Apple MPS + CPU with auto-detection |
| 12 | **Safe Loading** | `safe_load_checkpoint()` — PyTorch 2.6+ weights_only=True default |

### What's New in v2 (vs v1)

| Area | v1 | v2 |
|------|----|----|
| **Metric tracking** | Raw `float` variables | `Accumulator` class (d2l pattern) |
| **LR scheduling** | Config-only, no code | `CosineWarmupScheduler` — cosine decay + linear warmup |
| **Weight averaging** | Not available | `EMA` — exponential moving average for eval |
| **Optimizer setup** | Manual `torch.optim` calls | `create_optimizer()` factory (AdamW/SGD/Adam/RMSprop) |
| **Gradient clipping** | Hardcoded in train loop | `grad_clip()` utility + proper AMP unscale flow |
| **Gradient accumulation** | Not available | Built into `train_one_epoch()` |
| **Weight init** | Kaiming only | Full catalog: Kaiming, Xavier, Normal, per-activation guide |
| **Multi-GPU** | Not covered | DataParallel + DDP patterns |
| **Data augmentation** | Not covered | Train/val transform separation, RandAugment, anti-patterns |
| **Debugging** | Not covered | Anomaly detection, NaN checking, early stopping, CUDA errors |
| **Numerical stability** | Not covered | Label smoothing, eps handling, fp16 scaling, gradient norm monitoring |

### New imports in v2

```python
from src.reproducibility import (
    Accumulator,              # Clean metric tracking
    EMA,                      # Weight averaging for eval
    create_optimizer,         # Optimizer factory
    CosineWarmupScheduler,    # LR: warmup → cosine decay
    grad_clip,                # Gradient clipping utility
)
```

---

## Overview

This skill helps researchers make their deep learning projects fully reproducible for paper publication:

- Environment templates (requirements.txt, Dockerfile, environment.yml)
- Deterministic training for PyTorch and TensorFlow
- Production-grade training infrastructure (EMA, schedulers, gradient management)
- Standard project structure for research projects
- Type-safe configuration management
- Publication-ready documentation templates
- Debugging and numerical stability patterns

---

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

```bash
# Global installation
~/.claude/skills/dl-reproducibility-pack/

# Or project-local
your-project/.claude/skills/dl-reproducibility-pack/
```

---

## Usage

### Activating the Skill

Once installed, Claude Code will automatically activate this skill when you:

- Work on PyTorch or TensorFlow research projects
- Ask about reproducibility, random seeds, or determinism
- Need to create environment files or documentation

### Manual Activation

```
Please use the dl-reproducibility-pack skill to make my project reproducible.
```

---

## Components

### SKILL.md (19 topics)

The main skill definition covering:
- Model architecture, weight initialization, training loops
- Metric tracking, gradient management, LR scheduling
- EMA, multi-GPU training, debugging & diagnostics
- Data pipelines, augmentation, checkpointing
- Environment setup, configuration, documentation templates

### scripts/

- `reproducibility.py` — Core: seeds, device, Accumulator, EMA, CosineWarmupScheduler, train/eval loops with AMP + gradient accumulation
- `seed_worker.py` — DataLoader worker seeding, reproducible loader factory, batch samplers, collate utilities
- `config.py` — Type-safe YAML configuration with 6 dataclass-based sections (experiment, model, data, training, hardware, logging)

---

## Example: Full Training Pipeline (v2)

```python
from src.reproducibility import (
    set_seed, get_device,
    Accumulator, EMA, CosineWarmupScheduler,
    create_optimizer, grad_clip,
    train_one_epoch, evaluate,
)
from src.seed_worker import create_reproducible_dataloader

# Determinism
device = get_device()
set_seed(42)

# Data
train_loader = create_reproducible_dataloader(train_dataset, batch_size=128, seed=42)
val_loader = create_reproducible_dataloader(val_dataset, batch_size=256, seed=42, shuffle=False)

# Model + optimizer + scheduler + EMA
model = MyModel().to(device)
optimizer = create_optimizer(model, "AdamW", lr=0.001, weight_decay=0.0001)
scheduler = CosineWarmupScheduler(optimizer, warmup_steps=500, total_steps=10000)
ema = EMA(model, decay=0.999)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.amp.GradScaler("cuda")

# Training loop
for epoch in range(num_epochs):
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device,
        scaler=scaler,
        max_grad_norm=1.0,
        accumulation_steps=4,      # effective batch = 128 × 4 = 512
        ema=ema,
    )
    scheduler.step()

    # Validate with EMA weights
    ema.apply_shadow()
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    ema.restore()

    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
```

---

## License

MIT License

---

## Contributing

Contributions welcome! Please submit issues and pull requests on GitHub.
