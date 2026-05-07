---
name: dl-reproducibility-pack
description: Deep learning reproducibility toolkit v3.1 — 2025 AAAI/Nature reproducibility standards, PyTorch 2.7 patterns, FSDP2, experiment tracking, full RNG checkpointing.
origin: Custom + d2l-zh + AAAI 2025 Checklist + PyTorch 2.6/2.7 best practices
---

# Deep Learning Reproducibility Pack v3.1

A comprehensive toolkit combining 2025 reproducibility standards (AAAI/Nature checklists) with PyTorch 2.6/2.7 features for building robust, efficient, and fully reproducible deep learning research projects.

## Version Compatibility

All features auto-detect your environment and degrade gracefully.
Run `python scripts/compat.py` for a full compatibility report.

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Python | 3.8+ | 3.10+ | 3.9+ for modern type hints |
| PyTorch | 1.10+ | 2.4+ | Core features work on 1.10; compile needs 2.0; FSDP2 needs 2.4; Mega Cache needs 2.7 |
| CUDA | — | 11.8+ | CPU-only works but slower |
| torchmetrics | — | latest | Optional: standardized evaluation |
| scikit-learn | — | latest | Optional: `stratified_split()` |

**Feature availability by PyTorch version:**

| Feature | Min PyTorch | Fallback |
|---------|------------|----------|
| `set_seed` / `get_device` / `Accumulator` | 1.10 | — |
| `train_one_epoch` (AMP) | 1.10 | AMP disabled |
| `EMA` / `CosineWarmupScheduler` / `grad_clip` | 1.10 | — |
| `compile_model()` | 2.0 | Returns uncompiled model |
| `create_distributed_model("fsdp2")` | 2.4 | Auto-fallback to DDP |
| Mega Cache (`use_mega_cache=True`) | 2.7 | Silently skipped |
| `safe_load_checkpoint()` | 1.10 | weights_only=False if < 2.6 |
| MPS device (Apple Silicon) | 1.12 | Falls back to CPU |
| `lock_environment()` | 1.10 + pip | — |

## When to Activate

Activate when the user:
- Is preparing a deep learning paper for publication (PyTorch or TensorFlow)
- Wants to make DL code reproducible
- Needs to document training experiments
- Asks about random seeds, determinism, or reproducibility in DL
- Wants to create a standard research project structure
- Needs environment files (Dockerfile, requirements.txt, environment.yml)
- Wants to generate a reproducibility README for their DL project
- Needs to set up hyperparameter configuration management
- Wants to scaffold a complete reproducible research project
- Asks how to make their training code deterministic
- Needs to create a citation file for their research
- Writing new PyTorch models or training scripts
- Debugging training loops or data pipelines
- Optimizing GPU memory usage or training speed

## Core Principles

### 1. Reproducibility First

Always set all random seeds for reproducible results. This is the foundation of scientific deep learning.

```python
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    import random, numpy as np, torch, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.allow_tf32 = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

### 2. Device-Agnostic Code

Always write code that works on both CPU and GPU without hardcoding devices.

```python
# Good: Device-agnostic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
data = data.to(device)

# Bad: Hardcoded device (crashes if no GPU)
model = MyModel().cuda()
data = data.cuda()
```

### 3. Explicit Shape Management

Document tensor shapes in forward pass comments to catch shape mismatches early.

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (batch_size, channels, height, width)
    x = self.conv1(x)    # -> (batch_size, 32, H, W)
    x = self.pool(x)      # -> (batch_size, 32, H//2, W//2)
    x = x.view(x.size(0), -1)  # -> (batch_size, 32*H//2*W//2)
    return self.fc(x)    # -> (batch_size, num_classes)
```

---

## Topic Areas

### 1. Model Architecture Patterns

#### Clean nn.Module Structure

```python
# Good: Well-organized module with explicit initialization
class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64 * 16 * 16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

# Apply initialization
model = ImageClassifier(num_classes=100)
model.apply(model._init_weights)
```

#### Anti-patterns to Avoid

```python
# Bad: Everything in forward (creates weights each call!)
class BadModel(nn.Module):
    def forward(self, x):
        x = F.conv2d(x, weight=self.make_weight())  # Creates weight each call!
        return x

# Bad: In-place operations breaking autograd
x = F.relu(x, inplace=True)  # Can break gradient computation
x += residual             # In-place add breaks autograd graph

# Good: Out-of-place operations
x = F.relu(x)
x = x + residual
```

---

### 1b. Advanced Weight Initialization

Proper weight initialization prevents vanishing/exploding gradients.
Choose the init strategy based on activation function and architecture.

```python
def init_weights(model: nn.Module, mode: str = "kaiming") -> None:
    """Initialize model weights with the appropriate strategy."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if mode == "kaiming":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif mode == "xavier":
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            if mode == "xavier":
                nn.init.xavier_uniform_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)

model.apply(init_weights)
```

#### Initialization Quick Reference

| Module | Init | Rationale |
|--------|------|-----------|
| `nn.Conv2d` + ReLU | `kaiming_normal_(fan_out)` | Preserves variance through ReLU |
| `nn.Conv2d` + Tanh/Sigmoid | `xavier_uniform_` | Correct for symmetric activations |
| `nn.Linear` | `xavier_uniform_` | Default for fully-connected layers |
| `nn.BatchNorm2d` | `ones_(weight), zeros_(bias)` | Identity at initialization |
| `nn.Embedding` | `normal_(0, 1)` | Standard normal |
| `nn.LSTM/GRU` | `xavier_uniform_` | Helps gradient flow through recurrence |
| Transformer | `xavier_uniform_` or `normal_(0, 0.02)` | Small std for attention stability |

```python
# Anti-pattern: default init only (may cause training instability)
model = MyModel()  # Uses PyTorch defaults — varies by layer type

# Good: explicit init matching activation functions
model.apply(lambda m: nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if isinstance(m, (nn.Conv2d, nn.Linear)) else None)
```

---

### 2. Training Loop Patterns

#### Standard Training Loop

```python
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()  # Always set train mode
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        # Mixed precision training
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            output = model(data)
            loss = criterion(output, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

#### Validation Loop

```python
@torch.no_grad()  # More efficient than wrapping in torch.no_grad() block
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()  # Always set eval mode — disables dropout, uses running BN stats
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        total_loss += criterion(output, target).item()
        correct += (output.argmax(1) == target).sum().item()
        total += target.size(0)

    return total_loss / len(dataloader), correct / total
```

#### Anti-patterns

```python
# Bad: Forgetting model.eval() during validation
model.train()
with torch.no_grad():
    output = model(val_data)  # Dropout still active! BatchNorm uses batch stats!

# Good: Always set eval mode
model.eval()
with torch.no_grad():
    output = model(val_data)

# Bad: Moving data to GPU inside the training loop repeatedly
for data, target in dataloader:
    model = model.cuda()  # Moves model EVERY iteration!

# Good: Move model once before the loop
model = model.to(device)
for data, target in dataloader:
    data, target = data.to(device), target.to(device)

# Bad: Using .item() before backward
loss = criterion(output, target).item()  # Detaches from graph!
loss.backward()  # Error: can't backprop through .item()

# Good: Call .item() only for logging
loss = criterion(output, target)
loss.backward()
print(f"Loss: {loss.item():.4f}")  # .item() after backward is fine
```

---

### 2b. Metric Tracking (Accumulator)

Track metrics cleanly across batches with the `Accumulator` class used throughout d2l.

```python
class Accumulator:
    """Accumulate n variables — cleaner than raw scalars."""
    def __init__(self, n: int) -> None:
        self.data = [0.0] * n

    def add(self, *args) -> None:
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx: int) -> float:
        return self.data[idx]

# Usage: track loss sum, accuracy sum, sample count
metric = Accumulator(3)
for X, y in dataloader:
    metric.add(loss_val * X.shape[0], acc_val * X.shape[0], X.shape[0])
return metric[0] / metric[2], metric[1] / metric[2]
```

---

### 2c. Gradient Management

#### Gradient Clipping

Essential for RNN, Transformer, and deep network training stability.

```python
def grad_clip(model: nn.Module, max_norm: float) -> None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# Best practice: clip AFTER backward, BEFORE optimizer.step()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# With AMP: unscale before clipping
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

#### Gradient Accumulation

Simulate large batch sizes when GPU memory is limited.

```python
accumulation_steps = 4  # effective batch = batch_size × 4
optimizer.zero_grad(set_to_none=True)

for batch_idx, (data, target) in enumerate(dataloader):
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target) / accumulation_steps  # Normalize!
    loss.backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

---

### 2d. Learning Rate Scheduling

#### CosineAnnealing with Linear Warmup

The most effective general-purpose schedule (d2l sec_scheduler).

```python
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps,
                 base_lr=0.001, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.min_lr + (self.base_lr - self.min_lr) * \
                   self.current_step / max(1, self.warmup_steps)
        if self.current_step >= self.total_steps:
            return self.min_lr
        progress = (self.current_step - self.warmup_steps) / \
                   max(1, self.total_steps - self.warmup_steps)
        return self.min_lr + (self.base_lr - self.min_lr) * \
               (1 + math.cos(math.pi * progress)) / 2

    def step(self):
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.current_step += 1

# Usage: call scheduler.step() after optimizer.step() every batch
scheduler = CosineWarmupScheduler(optimizer, warmup_steps=500, total_steps=10000)
for batch in dataloader:
    loss.backward()
    optimizer.step()
    scheduler.step()
```

#### Other Built-in Schedulers

| Scheduler | When to Use |
|-----------|-------------|
| `StepLR` | Simple step decay every N epochs |
| `MultiStepLR` | Milestone-based decay (e.g., at epoch 30, 60, 80) |
| `CosineAnnealingLR` | Most CV tasks, smooth decay |
| `ReduceLROnPlateau` | Auto-reduce when validation metric plateaus |
| `OneCycleLR` | Fast training with super-convergence |
| `CosineAnnealingWarmRestarts` | Periodic warm restarts for ensembles |

#### Anti-patterns

```python
# Bad: Constant learning rate throughout training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(200):
    train(...)  # Never decays lr — suboptimal convergence

# Bad: Decaying too early (before warmup finishes)
scheduler.step()  # Called before model stabilizes

# Good: Warmup first, then cosine decay
scheduler = CosineWarmupScheduler(optimizer, warmup_steps=500, total_steps=10000)
```

---

### 2e. Model Weight Averaging (EMA)

Exponential Moving Average of model weights for better generalization.
Used in SSL, diffusion models, and competitive benchmarks.

```python
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        """Call before evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Call after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()

# Usage pattern
ema = EMA(model, decay=0.999)
for epoch in range(epochs):
    for batch in dataloader:
        loss.backward()
        optimizer.step()
        ema.update()          # Update EMA every step
    ema.apply_shadow()        # Switch to EMA weights
    val_metric = evaluate(model, val_loader)
    ema.restore()             # Switch back to training weights
```

---

### 2f. Multi-GPU Training

#### DataParallel (single-machine, quick setup)

```python
# Pattern from d2l train_ch13 — simplest path to multi-GPU
devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
model = nn.DataParallel(model, device_ids=devices).to(devices[0])

# Training loop — data goes to devices[0], DataParallel handles scatter/gather
for X, y in train_loader:
    X, y = X.to(devices[0]), y.to(devices[0])
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
```

#### DistributedDataParallel (multi-node, production)

```python
# Use DDP for serious multi-GPU — better performance than DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank: int, world_size: int):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# In training script:
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```

#### Device helpers (d2l pattern)

```python
def try_gpu(i: int = 0) -> torch.device:
    """Return gpu(i) if available, else cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus() -> list[torch.device]:
    """Return all available GPUs, or [cpu()] if none."""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
```

---

### 2g. Debugging & Diagnostics

#### Gradient Checking

```python
# Enable anomaly detection to catch NaNs and backward errors
torch.autograd.set_detect_anomaly(True)  # Only during debugging — slow!

# Check for NaN gradients
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN gradient in {name}")
```

#### Overfitting Detection

```python
# If train loss decreases but val loss increases → overfitting
# Solutions: more data, stronger augmentation, dropout, weight decay, early stopping

def should_early_stop(val_losses: list[float], patience: int = 10) -> bool:
    if len(val_losses) < patience:
        return False
    best_idx = val_losses.index(min(val_losses))
    return len(val_losses) - best_idx > patience
```

#### Numerical Stability

```python
# Use Label Smoothing to prevent overconfident predictions
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Avoid log(0) in custom losses — add eps
loss = -torch.log(pred + 1e-8)

# Use log_softmax + NLLLoss instead of softmax + log + NLLLoss
# PyTorch's CrossEntropyLoss already does this internally

# For fp16 training, use loss scaling
scaler = torch.amp.GradScaler("cuda")

# Monitor for exploding gradients
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
if total_norm > 100:
    print(f"Warning: large gradient norm {total_norm:.1f}")
```

#### Common CUDA Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `CUDA out of memory` | Batch too large / model too large | Reduce batch, use gradient checkpointing, `torch.cuda.empty_cache()` |
| `device-side assert triggered` | Label index out of range | Check `num_classes` matches label range |
| `CUDNN_STATUS_NOT_INITIALIZED` | Inconsistent CUDA/cuDNN versions | Match versions |
| `dtype mismatch` | Mixed fp16/fp32 tensors | Cast to consistent dtype before ops |

---

### 3. Data Pipeline Patterns

#### Custom Dataset

```python
class ImageDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        labels: dict[str, int],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.image_paths = list(Path(image_dir).glob("*.jpg"))
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[self.image_paths[idx].stem]
        if self.transform:
            img = self.transform(img)
        return img, label
```

#### Efficient DataLoader Configuration

```python
# Good: Optimized DataLoader for reproducibility
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,          # Faster CPU->GPU transfer
    persistent_workers=True,   # Keep workers alive between epochs
    drop_last=True,           # Consistent batch sizes for BatchNorm
    worker_init_fn=seed_worker,
    generator=torch.Generator().manual_seed(seed),
)

# Bad: Slow defaults (no reproducibility)
dataloader = DataLoader(dataset, batch_size=32)  # num_workers=0, no pin_memory
```

#### Custom Collate for Variable-Length Data

```python
def collate_fn(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, torch.tensor(labels)

dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

---

### 3b. Data Augmentation Patterns

Data augmentation is essential for generalization and implicitly enforces invariances.
Always validate that augmented outputs are still meaningful for your task.

```python
import torchvision.transforms as T

# Image classification — standard recipe
train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Test time — no augmentation, just resize + normalize
val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# RandAugment — strong augmentation for limited data (d2l sec_image_augmentation)
aug = T.RandAugment(num_ops=2, magnitude=9)

# CutMix / MixUp — advanced augmentation
# Use torchvision.transforms.v2.MixUp(alpha=1.0) or timm library
```

#### Augmentation Do's and Don'ts

```python
# Bad: Applying augmentation to validation set
val_dataset = MyDataset(transform=train_transform)  # Wrong!

# Good: Separate transforms for train/val
train_dataset = MyDataset(transform=train_transform)
val_dataset = MyDataset(transform=val_transform)

# Bad: Normalizing with wrong stats
T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Generic, suboptimal

# Good: Use dataset-specific statistics
# ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# Calculate your own: compute mean/std over training set
```

---

### 4. Checkpointing Patterns

#### Save and Load Checkpoints

```python
# Good: Complete checkpoint with all training state
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)

def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint

# Bad: Only saving model weights (can't resume training)
torch.save(model.state_dict(), "model.pt")

# Also Bad: Saving entire model (fragile, not portable)
torch.save(model, "model.pt")
```

---

### 5. Environment & Dependencies

#### Version Detection

```python
import torch, sys

def get_pytorch_environment():
    """Get comprehensive environment information for PyTorch."""
    return {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "gpu_memory_gb": [torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }
```

#### Environment Template Files

*requirements.txt for PyTorch*

```txt
# Core ML Framework (pin exact versions)
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2

# CUDA build must match torch version
# For torch==2.1.2 with CUDA 11.8

# Numerical computing
numpy==1.24.3
scipy==1.11.4

# Logging and experiment tracking
tensorboard==2.15.1
wandb==0.16.1

# Configuration management
pyyaml==6.0.1
omegaconf==2.3.0

# Data handling
pandas==2.1.4
pillow==10.1.0

# Development and testing
pytest==7.4.3
black==23.12.1
```

*environment.yml for Conda*

```yaml
name: dl-research
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.9.18
  - pip=23.3.1
  - pip:
    - torch==2.1.2
    - torchvision==0.16.2
    - numpy==1.24.3
    - scipy==1.11.4
    - tensorboard==2.15.1
    - wandb==0.16.1
    - pyyaml==6.0.1
    - omegaconf==2.3.0
```

*Dockerfile for complete containerization*

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.9=3.9.18-1~22.04 \
    python3-pip=22.3.1 \
    git && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash researcher
WORKDIR /home/researcher

COPY requirements.txt /home/researcher/
RUN pip install --no-cache-dir --user -r requirements.txt

COPY . /home/researcher/

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV CUDA_LAUNCH_BLOCKING=1

USER researcher

CMD ["python", "src/train.py", "--config", "configs/default.yaml"]
```

---

### 6. Random Seeds & Determinism

#### PyTorch Determinism

```python
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all random seeds for full reproducibility.
    """
    import random, numpy as np, torch, os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.allow_tf32 = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

#### TensorFlow Determinism

```python
def set_seed_tf(seed: int = 42) -> None:
    import tensorflow as tf, numpy as np, random, os

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.config.experimental.enable_op_determinism()
```

#### DataLoader Worker Seeding

```python
def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker process for reproducibility."""
    import numpy as np, random, torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

#### ReproducibilityContext

```python
class ReproducibilityContext:
    """Context manager for reproducible training sections."""
    def __init__(self, seed: int, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self.previous_state = {}

    def __enter__(self):
        import random, numpy as np, torch
        self.previous_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            self.previous_state["torch_cuda"] = torch.cuda.get_rng_state_all()
        set_seed(self.seed, self.deterministic)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import random, numpy as np, torch
        random.setstate(self.previous_state["python"])
        np.random.set_state(self.previous_state["numpy"])
        torch.set_rng_state(self.previous_state["torch"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.previous_state["torch_cuda"])

# Usage:
with ReproducibilityContext(seed=42, deterministic=True):
    results = train_epoch(model, train_loader)
```

---

### 7. Performance Optimization

#### Mixed Precision Training

```python
scaler = torch.amp.GradScaler("cuda")
for data, target in dataloader:
    with torch.amp.autocast("cuda"):
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

#### Gradient Checkpointing for Large Models

```python
from torch.utils.checkpoint import checkpoint

class LargeModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = checkpoint(self.block1, x, use_reentrant=False)
        x = checkpoint(self.block2, x, use_reentrant=False)
        return self.head(x)
```

#### torch.compile for Speed (PyTorch 2.0+)

```python
model = torch.compile(model, mode="reduce-overhead")
# Modes: "default" (safe), "reduce-overhead" (faster), "max-autotune" (fastest)
```

---

### 8. Project Structure

```
research_project/
├── configs/
│   ├── default.yaml             # Default experiment config
│   └── exp_*.yaml              # Specific experiment configs
│
├── data/
│   ├── __init__.py
│   ├── dataset.py              # Dataset class definitions
│   ├── transforms.py           # Data augmentation transforms
│   └── download_data.sh       # Data download script
│
├── src/
│   ├── __init__.py
│   ├── model.py               # Model architecture
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Evaluation functions
│   ├── reproducibility.py    # Seed, device, Accumulator, EMA, CosineWarmup
│   ├── config.py             # Configuration management
│   ├── seed_worker.py       # Reproducible DataLoader utilities
│   └── utils.py             # Helper utilities
│
├── logs/
│   ├── checkpoints/         # Model checkpoints
│   └── tensorboard/         # TensorBoard logs
│
├── scripts/
│   ├── train.sh             # Training launch script
│   ├── evaluate.sh          # Evaluation script
│   └── sweep.sh             # Hyperparameter sweep
│
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_reproducibility.py
│   └── test_data.py
│
├── environment.yml           # Conda environment
├── requirements.txt        # pip dependencies
├── Dockerfile              # Docker container
├── README.md               # Project documentation
├── CITATION.cff            # Citation metadata
└── .gitignore              # Git ignore rules
```

---

### 9. Configuration Templates

#### Default Config (configs/default.yaml)

```yaml
experiment:
  name: "default_experiment"
  seed: 42
  deterministic: true
  output_dir: "outputs/default"
  save_checkpoints: true
  resume_from: null

model:
  name: "ResNet50"
  pretrained: false
  num_classes: 1000
  dropout: 0.0

data:
  dataset: "CIFAR100"
  data_dir: "./data"
  image_size: 224
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  augmentation: true
  num_classes: 100
  val_split: 0.1
  test_split: 0.0
  split_seed: 42
  split_file: null

training:
  batch_size: 128
  num_epochs: 100
  learning_rate: 0.001
  min_lr: 0.0
  optimizer: "AdamW"
  scheduler: "CosineAnnealingLR"
  warmup_epochs: 5
  warmup_steps: 0
  weight_decay: 0.0001
  gradient_clip_norm: 1.0
  mixed_precision: false
  accumulation_steps: 1
  ema_decay: 0.0
  distributed_strategy: "ddp"
  compile_model: false
  compile_mode: "reduce-overhead"
  compile_dynamic: false
  use_mega_cache: false
  elastic_training: false

hardware:
  gpu_ids: [0]
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  device: "auto"
  world_size: 1
  local_rank: 0

logging:
  log_interval: 10
  eval_interval: 1
  save_checkpoint_interval: 10
  profiler_log_dir: "./profiler_logs"
  track_backend: "dummy"
  track_experiment_name: "experiment"
  use_tensorboard: true
  use_wandb: false
  wandb_project: "my-research"
```

---

### 10. Documentation Templates

#### README.md Structure

```markdown
# Project Title

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.9.18 |
| PyTorch | 2.1.2 |
| CUDA | 11.8 |

## Reproducibility

- Master seed: 42 (configurable via `configs/default.yaml`)
- Deterministic: true

## Usage

```bash
python src/train.py --config configs/default.yaml
```

## Results

| Model | Dataset | Metric | Value | Std |
|-------|---------|--------|-------|-----|
| ResNet50 | CIFAR-100 | Accuracy | 85.3% | 0.2% |
```

#### CITATION.cff

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
type: software
title: "Project Title"
authors:
  - family-names: "Author"
    given-names: "Name"
repository-code: "https://github.com/username/project"
license: MIT
```

---

### 11. Interactive Diagnostic Workflow (Plan-First)

**CRITICAL: When a user asks to make their project reproducible, you MUST follow this
diagnostic-first approach. Do NOT immediately generate code or templates.**
First understand the project, then propose a tailored plan, then implement.

#### Phase 1 — Diagnostic Questionnaire

Ask the following questions (adapt order based on conversation context).
**Stop after Phase 1 and present the diagnosis summary before proceeding.**

**A. Project Basics** (ask all that are unknown)

1. What deep learning framework? PyTorch / TensorFlow / JAX / other?
2. What domain? CV / NLP / tabular / RL / audio / multimodal / other?
3. What's the current state?
   - A) Just starting, no code yet
   - B) Have working code but no reproducibility measures
   - C) Partially reproducible, want to level up
   - D) Almost ready for paper submission
4. What hardware do you use?
   - A) Single NVIDIA GPU
   - B) Multi-GPU (same machine)
   - C) Multi-node cluster
   - D) Apple Silicon (MPS)
   - E) CPU only
   - F) Cloud (specify: AWS/GCP/Azure/etc.)

**B. Reproducibility Requirements** (ask all that are unknown)

5. What's the publication target?
   - A) Top conference (NeurIPS/ICML/CVPR/ACL etc.) — strictest requirements
   - B) Journal — standard requirements
   - C) Workshop / arXiv — moderate requirements
   - D) Internal / not publishing — minimal requirements
6. Do you need bit-for-bit exact reproducibility?
   - A) Yes — every run must produce identical weights
   - B) Within statistical tolerance — mean±std across seeds is sufficient
7. Do you need to resume training from checkpoints?
   - A) Yes, long training that may be interrupted
   - B) No, training is short enough to restart
8. How many random seeds for reporting?
   - A) 1 seed (exploratory)
   - B) 3 seeds (standard)
   - C) 5+ seeds (rigorous)
   - D) Not sure yet

**C. Data & Infrastructure** (ask all that are unknown)

9. What's the dataset situation?
   - A) Public benchmark dataset (CIFAR, ImageNet, etc.)
   - B) Private dataset (need to document preprocessing)
   - C) Both train/val/test split already defined
   - D) Need to create splits
10. Are you using experiment tracking?
    - A) No tracking yet
    - B) TensorBoard only
    - C) W&B / MLflow / Trackio
    - D) Want a recommendation
11. Do you have Docker available?
    - A) Yes, already using it
    - B) Docker installed but not using
    - C) No Docker — want help setting up
    - D) Can't use Docker (HPC restrictions, etc.)
12. What's your Python environment?
    - A) pip + virtualenv/venv
    - B) conda/mamba
    - C) pip only (global)
    - D) Poetry / uv / other

**D. Existing Code Check** (scan the project, then ask only if unclear)

13. Check for existing: `set_seed()`, `torch.manual_seed()`, `cudnn.deterministic` calls
14. Check for existing: `model.train()` / `model.eval()` patterns
15. Check for existing: checkpoint save/load code
16. Check for existing: DataLoader `worker_init_fn` or `generator`

---

#### Phase 2 — Diagnosis & Tailored Plan

After Phase 1, present a **diagnosis summary** in this format:

```markdown
## 项目诊断

| 维度 | 当前状态 | 建议 |
|------|----------|------|
| 框架 | PyTorch | — |
| 领域 | CV (图像分类) | — |
| 代码状态 | 有代码，无可复现性 | 需要注入种子设定 + 数据分割 |
| 硬件 | 单 GPU | DDP 不需要，AMP 建议开启 |
| 发表目标 | 顶会 (CVPR) | 需要最高级别可复现性 |
| 精确度要求 | 统计容忍度内 | 3 种子，mean±std 报告 |
| 续训 | 不需要 | 基础检查点即可 |
| 数据集 | 公开 (CIFAR100) | 保存预处理流程 |
| 实验追踪 | 无 | 推荐 Trackio (轻量) |
| Docker | 无 | 生成 Dockerfile 模板 |
| 环境 | pip + venv | pip freeze 完整锁定 |

**建议实施计划 (按优先级排序):**
1. [P0] 注入确定性种子 + DataLoader worker seeding
2. [P0] 锁定环境 (pip freeze + 环境信息)
3. [P0] 创建 train/val 分割并保存
4. [P1] 添加梯度裁剪 + CosineWarmup 调度器
5. [P1] 生成 requirements_frozen.txt + Dockerfile
6. [P2] 集成 ExperimentTracker (Trackio)
7. [P2] 二跑验证 CI 检查
8. [P3] AMP 混合精度 (可选，单 GPU 收益小)

回复 "执行" 开始实施以上计划，或告诉我需要调整的部分。
```

**Priority assignment rules:**
- **P0 (Must Have):** Seeds, environment lock, data splits — required by all checklists
- **P1 (Strongly Recommended):** Gradient mgmt, LR schedule, checkpoint, Docker — significant value
- **P2 (Nice to Have):** Tracking, profiling, verification CI — quality-of-life improvements
- **P3 (Optional):** AMP, FSDP2, torch.compile, TorchElastic — only if hardware/need justifies

**Customization rules based on answers:**
- `publication=top_conf` → Require P0+P1+P2, suggest P3
- `publication=internal` → Only P0 is mandatory
- `hardware=cpu_only` → Skip AMP, FSDP2, CUDA-specific settings
- `hardware=apple_silicon` → Use MPS device, skip CUDA settings
- `domain=nlp` → Add gradient clipping (critical for RNN/Transformer)
- `domain=cv` → Add data augmentation patterns
- `resume_training=yes` → Require full RNG checkpoint (save_full_checkpoint)
- `bit_exact=yes` → Set `torch.use_deterministic_algorithms(True)`, warn about 10-30% slowdown
- `framework=tensorflow` → Use TF-specific `set_seed_tf()`, skip torch.compile
- `dataset=private` → Add DatasetVersioning + preprocessing documentation
- `docker=no_cant` → Generate conda environment.yml as alternative

---

#### Phase 3 — Implementation

Once user approves the plan, implement in this order:

1. **Environment**: `lock_environment()` → `requirements_frozen.txt` → Dockerfile/conda env
2. **Determinism**: Inject `set_seed()` at top of training script, add `seed_worker()` to DataLoader
3. **Data**: Create/save dataset splits, add `DatasetVersioning` if private data
4. **Training loop**: Add gradient clipping, CosineWarmupScheduler, Accumulator
5. **Checkpointing**: Replace `torch.save(model.state_dict())` with `save_full_checkpoint()`
6. **Evaluation**: Replace hand-rolled metrics with TorchMetrics via `get_metrics()`
7. **Tracking**: Add `ExperimentTracker` with appropriate backend
8. **Verification**: Add `verify_reproducibility()` to test suite
9. **Documentation**: Generate README + CITATION.cff + environment report

**For each step, show a diff preview** before applying changes (respecting user preference for confirmation).

---

#### Quick-Activation Shortcuts

If user explicitly asks for a specific feature without the full diagnostic:

```
"帮我锁定环境"        → lock_environment("env_lock")
"帮我加二跑验证"      → inject verify_reproducibility() into test suite
"帮我做数据集版本控制" → inject save_dataset_split() + DatasetVersioning
"帮我加实验追踪"      → inject ExperimentTracker with recommended backend
```

But if the request is broad ("make my project reproducible"), **always go through Phase 1→2→3**.

---

### 12. v3 Advanced Features (2025 PyTorch Best Practices)

### 12a. Full RNG State Checkpointing

Save ALL random number generator states for bit-exact training resumption.
Required by AAAI 2025 Reproducibility Checklist.

```python
# Save: model + optimizer + scheduler + Python/Numpy/PyTorch/CUDA RNG
save_full_checkpoint(
    model, optimizer, "checkpoint_epoch10.pt",
    epoch=10, global_step=5000, loss=train_loss,
    scheduler=scheduler, ema=ema, scaler=scaler,
)

# Load: restores EVERYTHING for exact continuation
ckpt = load_full_checkpoint(
    "checkpoint_epoch10.pt", model, optimizer,
    scheduler=scheduler, ema=ema, scaler=scaler,
)
print(f"Resumed from epoch {ckpt['epoch']}, step {ckpt['global_step']}")
```

---

### 12b. Two-Run Reproducibility Verification

Nature 2025 checklist: run twice with the same seed, assert bit-identical loss curves.

```python
def train_with_seed(seed):
    set_seed(seed)
    losses = []
    for epoch in range(5):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        losses.append(loss)
    return losses

# Verify: should print PASS
assert verify_reproducibility(train_with_seed, seed=42)

# CI/CD guard: add to your test suite
def test_reproducibility():
    assert verify_reproducibility(train_with_seed, seed=0)
```

---

### 12c. Complete Environment Locking

Capture everything needed to reproduce the exact environment.

```python
# Generates 3 files:
#   env_lock/requirements_frozen.txt  — ALL packages with pinned versions
#   env_lock/environment_info.json     — PyTorch, CUDA, GPU details
#   env_lock/system_info.txt           — OS, kernel, nvidia-smi
lock_environment("env_lock")
```

---

### 12d. Dataset Versioning & Split Management

Pre-define splits and version datasets for reproducible data pipelines.

```python
from src.seed_worker import (
    save_dataset_split, load_dataset_split,
    stratified_split, DatasetVersioning,
)

# 1. Create and save reproducible stratified split
split = stratified_split(labels, val_ratio=0.1, test_ratio=0.1, seed=42)
save_dataset_split(split, "splits/fold_0.json",
                   metadata={"dataset": "CIFAR100", "seed": 42})

# 2. Record dataset version metadata
versioning = DatasetVersioning("cifar100", "v2")
versioning.record(num_samples=50000, augmentations="RandAugment")
versioning.save("data/version.json")

# 3. Reload exact same split later
indices = load_dataset_split("splits/fold_0.json")
train_dataset = Subset(dataset, indices["train"])
```

---

### 12e. FSDP2 Distributed Training

PyTorch 2.4+ FSDP2 is the 2025 standard for distributed training.
Uses `fully_shard()` with DTensor — 30% simpler API than FSDP1.

```python
# Single-machine multi-GPU
model = create_distributed_model(model, strategy="ddp")

# FSDP2 for large models (PyTorch 2.4+)
model = create_distributed_model(model, strategy="fsdp2")

# With custom FSDP2 mixed precision policy
from torch.distributed.fsdp import MixedPrecisionPolicy
model = create_distributed_model(
    model, strategy="fsdp2",
    mixed_precision=MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    ),
)
```

---

### 12f. Advanced torch.compile (PyTorch 2.6/2.7)

Leverage the latest compilation features for 30-50% speedup.

```python
# Basic: reduce-overhead mode (good for training)
model = compile_model(model, mode="reduce-overhead")

# Advanced: full-graph + dynamic shapes + Mega Cache (2.7+)
model = compile_model(
    model,
    mode="max-autotune",
    dynamic=True,         # Variable-length inputs
    fullgraph=True,       # Fail if any op can't be compiled
    use_mega_cache=True,  # Cross-machine portable cache (2.7+)
    cache_dir="./compile_cache",
)
```

PyTorch 2.6/2.7 features leveraged:
- `torch.compiler.set_stance("eager_on_recompile")` — fallback to eager on cache miss
- **Mega Cache** — `save_cache_artifacts()` / `load_cache_artifacts()` for cross-machine reuse
- **Prologue Fusion** — matmul preamble fused into kernel (automatic in 2.6+)

---

### 12g. Profiling & Benchmarking

Diagnose training bottlenecks with PyTorch Profiler.

```python
from src.profiling import ProfileContext, BenchmarkTimer, benchmark_model

# 1. Schedule-based profiling (warmup → active → wait)
with ProfileContext(log_dir="./profiler_logs", active=3) as prof:
    for batch in train_loader:
        loss.backward()
        optimizer.step()
        prof.step()
# Open: tensorboard --logdir=./profiler_logs

# 2. Quick CUDA-synchronized micro-benchmark
timer = BenchmarkTimer()
with timer:
    output = model(data)
print(f"Forward: {timer.elapsed_ms:.2f}ms")

# 3. Full throughput + memory benchmark
report = benchmark_model(model, input_shape=(128, 3, 224, 224), device=device)
print(f"Throughput: {report['throughput_imgs_per_sec']:.1f} imgs/sec")
print(f"GPU Memory: {report['gpu_memory_mb']:.1f} MB")
```

---

### 12h. TorchMetrics Integration

Standardized metrics eliminate implementation variance across papers.

```python
from src.reproducibility import get_metrics

# Create TorchMetrics collection — consistent across all experiments
metrics = get_metrics(
    ["Accuracy", "F1Score", "AUROC", "Precision", "Recall"],
    num_classes=100,
)

# Use in evaluation loop
for preds, targets in val_loader:
    for m in metrics.values():
        m.update(preds, targets)

results = {name: m.compute().item() for name, m in metrics.items()}
# {'Accuracy': 0.853, 'F1Score': 0.841, 'AUROC': 0.972, ...}
```

---

### 12i. Experiment Tracking

Unified interface across Trackio, MLflow, and W&B.

```python
from src.tracking import ExperimentTracker

# Trackio (HuggingFace, July 2025) — lightweight, open-source, local-first
tracker = ExperimentTracker("trackio", "resnet50_cifar100")
tracker.log_hyperparams(config.__dict__)

# MLflow — enterprise standard, self-hosted
tracker = ExperimentTracker("mlflow", "resnet50", tracking_uri="http://mlflow:5000")

# W&B — cloud-hosted, rich visualizations
tracker = ExperimentTracker("wandb", "resnet50", project="my-research")

for epoch in range(epochs):
    tracker.log_metrics({"train_loss": loss, "val_acc": acc}, step=epoch)
tracker.finish()
```

---

### 12j. TorchElastic Fault-Tolerant Training

Automatically recover from node failures during long training runs.

```python
from src.reproducibility import get_elastic_environment

env = get_elastic_environment()
if env["is_elastic"]:
    print(f"Rank {env['rank']}/{env['world_size']}, "
          f"Restart #{env['restart_count']}")
    # Load checkpoint on restart
    if env["restart_count"] > 0:
        load_full_checkpoint("latest.pt", model, optimizer)

# Launch with:
# torchrun --nproc_per_node=4 --max_restarts=3 train.py
```

---

### 12k. Multi-Platform Hardware Support

Automatic device selection across CUDA, Apple MPS, and CPU.

```python
# get_device() now supports MPS (Apple Silicon) + CUDA + CPU
device = get_device()  # cuda > mps > cpu

# Explicit GPU selection (d2l pattern)
device = try_gpu(0)          # gpu:0 or cpu
devices = try_all_gpus()     # [gpu:0, gpu:1, ...] or [cpu]

# Config-driven:
config.hardware.device = "mps"    # Force Apple Silicon
config.hardware.device = "cuda:1" # Force specific GPU
```

Supported platforms:
| Platform | Detection | PyTorch Version |
|----------|-----------|-----------------|
| NVIDIA CUDA | `torch.cuda.is_available()` | 1.0+ |
| Apple MPS | `torch.backends.mps.is_available()` | 1.12+ |
| CPU | Always available | All |

---

### 12l. Safe Model Loading

Secure checkpoint loading with PyTorch 2.6+ defaults.

```python
from src.reproducibility import safe_load_checkpoint

# Safe loading (weights_only=True by default in PyTorch 2.6+)
state = safe_load_checkpoint("model.pt", model)

# Full checkpoint (RNG states) — weights_only=False with warning
ckpt = load_full_checkpoint("full_checkpoint.pt", model, optimizer)
# Note: full checkpoints require weights_only=False because they contain
# non-tensor RNG state objects. Only load from trusted sources.
```

---

## Quick Reference: PyTorch Idioms

| Idiom | Description |
|-------|-------------|
| `model.train()` / `model.eval()` | Always set mode before train/eval — disables dropout, uses running BN |
| `torch.no_grad()` | Disable gradients for inference |
| `optimizer.zero_grad(set_to_none=True)` | More efficient gradient clearing (sets to None vs 0) |
| `.to(device)` | Device-agnostic tensor/model placement |
| `torch.amp.autocast("cuda")` | Mixed precision for ~2x speed on Ampere+ GPUs |
| `pin_memory=True` | Faster CPU→GPU data transfer |
| `persistent_workers=True` | Keep DataLoader workers alive between epochs |
| `torch.compile(mode="reduce-overhead")` | JIT compilation for speed (PyTorch 2.0+) |
| `loss / accumulation_steps` | Normalize loss when accumulating gradients |
| `torch.nn.utils.clip_grad_norm_` | Prevent exploding gradients (RNN/Transformer) |
| `weights_only=True` | Secure checkpoint loading (prevents code injection) |
| `torch.manual_seed(seed)` | Reproducible experiments |
| `torch.Generator().manual_seed(seed)` | Reproducible DataLoader shuffling |
| `torch.use_deterministic_algorithms(True)` | Full determinism (slower, but bit-identical runs) |
| `gradient_checkpointing` | Trade compute for memory in large models |
| `nn.utils.rnn.pad_sequence` | Pad variable-length sequences in collate_fn |
| `torch.backends.cudnn.benchmark=False` | Disable auto-tuning for determinism |
| `torch.backends.cudnn.allow_tf32=False` | Disable TF32 for bit-identical results |
| `label_smoothing` | Prevent overconfident predictions (better calibration) |
| `CosineWarmupScheduler` | Cosine LR decay with linear warmup — best general schedule |
| `save_full_checkpoint()` / `load_full_checkpoint()` | Save/restore ALL RNG states for bit-exact resume |
| `verify_reproducibility()` | Two-run identity test — CI guard for determinism |
| `lock_environment()` | Capture pip freeze + nvidia-smi + hardware info |
| `save_dataset_split()` / `load_dataset_split()` | Pre-defined reproducible data splits |
| `create_distributed_model(strategy="fsdp2")` | FSDP2 with fully_shard (PyTorch 2.4+) |
| `compile_model(use_mega_cache=True)` | Advanced torch.compile with Mega Cache (PyTorch 2.7) |
| `ProfileContext` / `benchmark_model()` | Profiler with warmup/active/wait scheduling |
| `get_metrics(["Accuracy","F1Score"])` | TorchMetrics — standardized evaluation |
| `ExperimentTracker("trackio")` | Unified tracking (Trackio/MLflow/W&B) |
| `get_elastic_environment()` | Detect TorchElastic for fault-tolerant training |
| `get_device(allow_mps=True)` | Multi-platform: CUDA > MPS > CPU |
| `safe_load_checkpoint(path, model)` | Secure loading with PyTorch 2.6+ weights_only default |

---

## Best Practices

### Do's

- **Always set all random seeds** — Python, NumPy, PyTorch/TensorFlow, and CUDA
- **Use `persistent_workers=True`** in DataLoader for faster and more reproducible training
- **Pin exact package versions** in requirements.txt using `==`
- **Document GPU model and driver version** — Different hardware can produce different results
- **Set `cudnn.benchmark = False`** for reproducibility-critical code
- **Run multiple seeds** and report mean±std — Single seed runs are not reliable
- **Use `torch.cuda.manual_seed_all()`** for multi-GPU training
- **Save checkpoints with optimizer state** for resumability
- **Use gradient clipping** (`max_norm=1.0`) for RNN, Transformer, and deep networks
- **Warm up the learning rate** for 5-10% of total steps to prevent early divergence
- **Use CosineAnnealing with warmup** — the best general-purpose LR schedule
- **Apply EMA** for the best validation metrics in competitive benchmarks
- **Select weight init based on activation**: Kaiming for ReLU, Xavier for Tanh/Sigmoid
- **Use Accumulator class** for clean metric tracking — cleaner than raw float variables
- **Normalize loss by `accumulation_steps`** when using gradient accumulation
- **Save full RNG state in checkpoints** — Python, NumPy, PyTorch, and CUDA RNG
- **Run two-run verification** — Assert identical loss curves with same seed as a CI check
- **Lock the full environment** — pip freeze ALL packages, not just direct dependencies
- **Pre-define and save dataset splits** — Never rely on random split at runtime
- **Use TorchMetrics for evaluation** — Eliminates implementation variance across papers
- **Profile before scaling** — Use ProfileContext to identify bottlenecks first
- **Use FSDP2 for large models** — 30% simpler API than FSDP1, better memory
- **Prefer torch.compile with Mega Cache** — Cross-machine portable compilation cache (2.7+)
- **Track experiments** — Use ExperimentTracker with Trackio/MLflow/W&B from day one
- **Use `weights_only=True` for model-only loading** — Default in PyTorch 2.6+

### Don'ts

- **Don't rely on a single run** — Report results across multiple seeds
- **Don't use `cudnn.benchmark = True`** in reproducibility-critical code
- **Don't leave workers unseeded** — Use `worker_init_fn=seed_worker`
- **Don't save entire model** — Use `model.state_dict()`
- **Don't use `.item()` before `backward()`** — Detaches from computation graph
- **Don't move model to GPU inside training loop** — Move once before the loop
- **Don't apply data augmentation to validation data**
- **Don't use default weight initialization** without considering activation functions
- **Don't decay learning rate before warmup completes** — Model needs time to stabilize
- **Don't forget `model.eval()` before validation** — Dropout/BatchNorm behavior differs
- **Don't save checkpoints without RNG states** — Makes exact resumption impossible
- **Don't hand-roll metrics** — Use TorchMetrics for standardized, peer-reviewed implementations
- **Don't load untrusted checkpoints with `weights_only=False`** — Security risk

---

## Common Pitfalls

```python
# Pitfall 1: Forgetting multi-GPU seed
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For all GPUs

# Pitfall 2: DataLoader without generator
g = torch.Generator()
g.manual_seed(seed)
loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=g)

# Pitfall 3: TF32 on Ampere GPUs
torch.backends.cudnn.allow_tf32 = False  # For reproducibility

# Pitfall 4: Missing RNG in checkpoint — can't resume exactly
torch.save({"model": model.state_dict()}, "ckpt.pt")  # Incomplete!

# Pitfall 5: Random split at every run — different data each time
train_idx, val_idx = train_test_split(...)  # No fixed seed!

# Pitfall 6: Hand-rolled metric vs TorchMetrics — different results
accuracy = (preds.argmax(1) == targets).float().mean()  # May differ from TorchMetrics

# Pitfall 7: Non-deterministic DataLoader without generator
loader = DataLoader(dataset, shuffle=True)  # Different order each run!

# Pitfall 8: Not capturing pip transitive dependencies
# requirements.txt lists only direct deps — different sub-dependency versions
```

---

*This skill combines reproducibility best practices with idiomatic PyTorch patterns for research-grade deep learning projects.*
