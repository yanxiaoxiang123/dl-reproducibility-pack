"""
reproducibility.py - Deep Learning Reproducibility & PyTorch Patterns

A unified module combining reproducibility utilities with idiomatic PyTorch patterns
for building robust, efficient, and reproducible deep learning projects.

Usage:
    from src.reproducibility import set_seed, get_device, train_one_epoch

    device = get_device()
    set_seed(42, deterministic=True)
    loss = train_one_epoch(model, dataloader, optimizer, criterion, device)

Author: dl-reproducibility-skills
"""

from __future__ import annotations

import json
import math
import random
import os
import platform
import sys
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader
    from torch import nn

# Lazy imports to avoid hard dependencies
def _get_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None

def _get_numpy():
    try:
        import numpy as np
        return np
    except ImportError:
        return None


def get_device(allow_mps: bool = True) -> "torch.device":
    """
    Get the best available device with multi-platform support.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Args:
        allow_mps: Whether to use Apple MPS if available

    Returns:
        torch.device: Best available device

    Example:
        >>> device = get_device()
        >>> print(device)
        cuda:0  # or mps:0 on Apple Silicon, or cpu
    """
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_mps and _torch_version() >= (1, 12) and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def try_gpu(i: int = 0) -> "torch.device":
    """Return gpu(i) if available, else cpu(). d2l pattern."""
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required")
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus() -> list["torch.device"]:
    """Return all available GPUs, or [cpu()] if none. d2l pattern."""
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required")
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def set_seed(
    seed: int = 42,
    framework: str = "pytorch",
    deterministic: bool = True,
) -> None:
    """
    Set all random seeds for reproducible deep learning experiments.

    Args:
        seed: The random seed to use
        framework: "pytorch" or "tensorflow"
        deterministic: If True, use fully deterministic algorithms (PyTorch only)

    Example:
        >>> set_seed(42, framework="pytorch", deterministic=True)
        >>> set_seed(42, framework="tensorflow")
    """
    np = _get_numpy()
    if np is None:
        raise ImportError("numpy is required for reproducibility")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if framework.lower() == "pytorch":
        _set_pytorch_seed(seed, deterministic)
    elif framework.lower() in ("tensorflow", "tf"):
        _set_tensorflow_seed(seed)
    else:
        raise ValueError(f"Unknown framework: {framework}. Use 'pytorch' or 'tensorflow'")


def _set_pytorch_seed(seed: int, deterministic: bool = True) -> None:
    """Set seeds for PyTorch reproducibility."""
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required for PyTorch reproducibility")

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.allow_tf32 = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _set_tensorflow_seed(seed: int) -> None:
    """Set seeds for TensorFlow 2.x reproducibility."""
    tf = _get_tf()
    if tf is None:
        raise ImportError("tensorflow is required for TensorFlow reproducibility")

    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, ValueError):
        pass


def get_environment_info(framework: str = "pytorch") -> dict:
    """
    Get comprehensive environment information for debugging and reproducibility records.

    Args:
        framework: "pytorch" or "tensorflow"

    Returns:
        Dictionary containing environment version information

    Example:
        >>> info = get_environment_info("pytorch")
        >>> print(info["pytorch_version"])
        2.1.2
    """
    torch = _get_torch()
    tf = _get_tf()
    np = _get_numpy()

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "random_seed_env": os.environ.get("PYTHONHASHSEED", "not set"),
    }

    if np:
        info["numpy_version"] = np.__version__

    if framework.lower() == "pytorch" and torch:
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_count": torch.cuda.device_count(),
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "gpu_memory_gb": [torch.cuda.get_device_properties(i).total_memory / 1e9
                                 for i in range(torch.cuda.device_count())],
            })
    elif framework.lower() in ("tensorflow", "tf") and tf:
        info["tensorflow_version"] = tf.__version__
        info["cuda_available"] = tf.test.is_gpu_available()
        if tf.test.is_gpu_available():
            info["gpu_devices"] = [d.name for d in tf.config.list_physical_devices('GPU')]

    return info


def check_determinism(framework: str = "pytorch") -> dict:
    """
    Check current determinism settings and report any potential issues.

    Args:
        framework: "pytorch" or "tensorflow"

    Returns:
        Dictionary containing determinism status and warnings
    """
    torch = _get_torch()

    status = {
        "framework": framework,
        "is_deterministic": True,
        "warnings": [],
        "settings": {},
    }

    if framework.lower() == "pytorch" and torch:
        cudnn_benchmark = torch.backends.cudnn.benchmark
        cudnn_deterministic = torch.backends.cudnn.deterministic

        status["settings"]["cudnn_benchmark"] = cudnn_benchmark
        status["settings"]["cudnn_deterministic"] = cudnn_deterministic

        if cudnn_benchmark:
            status["warnings"].append("cudnn.benchmark is True - may cause non-deterministic results")
            status["is_deterministic"] = False

        if not cudnn_deterministic:
            status["warnings"].append("cudnn.deterministic is False - may cause non-deterministic results")
            status["is_deterministic"] = False

        try:
            torch.use_deterministic_algorithms(True)
            status["settings"]["use_deterministic_algorithms"] = True
        except RuntimeError:
            status["settings"]["use_deterministic_algorithms"] = False
            status["warnings"].append("Some operations lack deterministic implementations")
            status["is_deterministic"] = False

    return status


def log_environment_info(filepath: Optional[str] = None, framework: str = "pytorch") -> dict:
    """
    Log environment information to a file for reproducibility records.

    Args:
        filepath: Path to write the log file. If None, only returns the info dict.
        framework: "pytorch" or "tensorflow"

    Returns:
        Dictionary containing environment information
    """
    info = get_environment_info(framework)
    info["seed_check"] = check_determinism(framework)

    if filepath:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            f.write("Deep Learning Environment Information\n")
            f.write("=" * 50 + "\n\n")
            for key, value in info.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"{key}: {value}\n")

    return info


# Training loop patterns following d2l / pytorch-patterns idioms
def train_one_epoch(
    model: "nn.Module",
    dataloader: "DataLoader",
    optimizer: "torch.optim.Optimizer",
    criterion: "nn.Module",
    device: "torch.device",
    scaler: Optional["torch.amp.GradScaler"] = None,
    max_grad_norm: Optional[float] = None,
    accumulation_steps: int = 1,
    ema: Optional[EMA] = None,
) -> float:
    """Train for one epoch following idiomatic PyTorch patterns.

    Args:
        model: The neural network model
        dataloader: Training data loader
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to train on
        scaler: Optional GradScaler for mixed precision
        max_grad_norm: Gradient clipping threshold (None = skip)
        accumulation_steps: Simulate larger batch via gradient accumulation
        ema: Optional EMA instance for weight averaging

    Returns:
        Average loss for the epoch
    """
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required")

    model.train()
    metric = Accumulator(2)  # loss_sum, sample_count
    optimizer.zero_grad(set_to_none=True)
    total_batches = len(dataloader)

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        window_start = (batch_idx // accumulation_steps) * accumulation_steps
        window_end = min(window_start + accumulation_steps, total_batches)
        accumulation_divisor = window_end - window_start

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            output = model(data)
            loss = criterion(output, target) / accumulation_divisor

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_accumulation_boundary = (batch_idx + 1) % accumulation_steps == 0
        is_last_batch = (batch_idx + 1) == len(dataloader)
        if is_accumulation_boundary or is_last_batch:
            if max_grad_norm is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update()

        metric.add(loss.item() * accumulation_divisor * data.size(0), data.size(0))

    return metric[0] / metric[1]


def evaluate(
    model: "nn.Module",
    dataloader: "DataLoader",
    criterion: "nn.Module",
    device: "torch.device",
) -> tuple[float, float]:
    """
    Evaluate the model following idiomatic PyTorch patterns.

    Args:
        model: The neural network model
        dataloader: Evaluation data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required")

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_size = target.size(0)
            total_loss += criterion(output, target).item() * batch_size
            correct += (output.argmax(1) == target).sum().item()
            total += batch_size

    return total_loss / total, correct / total


# For internal use when torch is available
def _get_tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        return None


# ── Metric tracking ──────────────────────────────────────────────

class Accumulator:
    """Accumulate values over n variables for clean metric tracking.

    Pattern from d2l — used throughout training loops to track
    loss, accuracy, and sample counts across batches.

    >>> metric = Accumulator(3)  # train_loss, train_acc, num_samples
    >>> metric.add(loss_val, acc_val, num_samples)
    >>> loss_avg = metric[0] / metric[2]
    """
    def __init__(self, n: int) -> None:
        self.data = [0.0] * n

    def add(self, *args) -> None:
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self) -> None:
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx: int) -> float:
        return self.data[idx]


# ── Exponential Moving Average ───────────────────────────────────

class EMA:
    """Exponential Moving Average of model parameters.

    Maintains shadow copies of parameters updated with exponential decay.
    Apply shadow weights before evaluation for better generalization
    (common in SSL, diffusion models, and large-scale training).

    >>> ema = EMA(model, decay=0.999)
    >>> for epoch in range(epochs):
    ...     train_one_epoch(model, loader, opt, crit, device, ema=ema)
    ...     ema.apply_shadow()
    ...     eval_metrics = evaluate(model, val_loader, crit, device)
    ...     ema.restore()
    """
    def __init__(self, model: "nn.Module", decay: float = 0.999) -> None:
        torch = _get_torch()
        if torch is None:
            raise ImportError("torch is required")
        self.model = model
        self.decay = decay
        self.shadow: dict[str, "torch.Tensor"] = {}
        self.backup: dict[str, "torch.Tensor"] = {}
        self._register()

    def _register(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self) -> None:
        """Replace model parameters with EMA shadow (call before eval)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original parameters (call after eval)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()


# ── Optimizer & scheduler factories ──────────────────────────────

def create_optimizer(
    model: "nn.Module",
    optimizer_name: str = "AdamW",
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    momentum: float = 0.9,
    betas: tuple[float, float] = (0.9, 0.999),
) -> "torch.optim.Optimizer":
    """Create optimizer by name. Covers the four most common choices."""
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required")
    name = optimizer_name.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup (d2l sec_scheduler pattern).

    >>> scheduler = CosineWarmupScheduler(optimizer, warmup_steps=500, total_steps=10000)
    >>> for batch in train_loader:
    ...     loss.backward()
    ...     optimizer.step()
    ...     scheduler.step()  # call per batch after optimizer.step()
    """
    def __init__(
        self,
        optimizer: "torch.optim.Optimizer",
        warmup_steps: int,
        total_steps: int,
        base_lr: float = 0.001,
        min_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0
        self._last_lr = base_lr

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.min_lr + (self.base_lr - self.min_lr) * self.current_step / max(1, self.warmup_steps)
        if self.current_step >= self.total_steps:
            return self.min_lr
        progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2

    def step(self) -> float:
        self._last_lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._last_lr
        self.current_step += 1
        return self._last_lr

    def state_dict(self) -> dict:
        return {'current_step': self.current_step}

    def load_state_dict(self, state: dict) -> None:
        self.current_step = state['current_step']


def grad_clip(model: "nn.Module", max_norm: float) -> None:
    """Clip gradient norm (essential for RNN/Transformer training stability)."""
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required")
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


# ── Improvement #1: Full RNG Checkpoint ────────────────────────

def save_full_checkpoint(
    model: "nn.Module",
    optimizer: "torch.optim.Optimizer",
    path: str,
    epoch: int = 0,
    global_step: int = 0,
    loss: float = 0.0,
    scheduler: Optional[object] = None,
    ema: Optional[EMA] = None,
    scaler: Optional["torch.amp.GradScaler"] = None,
) -> None:
    """Save complete training state including ALL RNG states for bit-exact resume.

    Per AAAI 2025 Reproducibility Checklist and PyTorch 2025 best practices,
    this saves Python/NumPy/PyTorch/CUDA RNG states alongside model state.
    """
    torch = _get_torch()
    np = _get_numpy()
    if torch is None or np is None:
        raise ImportError("torch and numpy are required")

    checkpoint: dict[str, object] = {
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()

    if scheduler is not None and hasattr(scheduler, 'state_dict'):
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if ema is not None:
        checkpoint["ema_shadow"] = {k: v.clone() for k, v in ema.shadow.items()}

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(checkpoint, path)


def load_full_checkpoint(
    path: str,
    model: "nn.Module",
    optimizer: "torch.optim.Optimizer",
    scheduler: Optional[object] = None,
    ema: Optional[EMA] = None,
    scaler: Optional["torch.amp.GradScaler"] = None,
) -> dict:
    """Load complete training state and restore ALL RNG states.

    Returns the checkpoint dict for accessing epoch, global_step, loss.
    """
    torch = _get_torch()
    np = _get_numpy()
    if torch is None or np is None:
        raise ImportError("torch and numpy are required")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    # NOTE: weights_only=False is required here because the checkpoint contains
    # non-tensor objects (RNG states). The saved file is under user control.

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore RNG states for bit-exact continuation
    random.setstate(checkpoint["python_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])

    if "cuda_rng" in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        if hasattr(scheduler, 'load_state_dict'):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if ema is not None and "ema_shadow" in checkpoint:
        for k, v in checkpoint["ema_shadow"].items():
            if k in ema.shadow:
                ema.shadow[k].copy_(v)

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


# ── Improvement #2: Two-Run Reproducibility Verification ──────

def verify_reproducibility(
    train_fn: object,
    seed: int = 42,
    tolerance: float = 1e-7,
) -> bool:
    """Run training twice with identical seed and assert loss curves match.

    Per Nature 2025 reproducibility checklist: two identical runs must produce
    bit-identical results when all sources of non-determinism are controlled.

    Args:
        train_fn: Callable(seed) -> list[float] that returns loss history
        seed: Random seed for both runs
        tolerance: Maximum allowed difference between corresponding losses

    Returns:
        True if runs are identical within tolerance
    """
    print(f"Verifying reproducibility with seed={seed}...")
    losses_run1 = train_fn(seed)
    losses_run2 = train_fn(seed)

    if len(losses_run1) != len(losses_run2):
        print(f"FAIL: Different number of steps ({len(losses_run1)} vs {len(losses_run2)})")
        return False

    max_diff = 0.0
    for i, (l1, l2) in enumerate(zip(losses_run1, losses_run2)):
        diff = abs(l1 - l2)
        max_diff = max(max_diff, diff)
        if diff > tolerance:
            print(f"FAIL: Step {i} differs by {diff:.2e} (l1={l1:.6f}, l2={l2:.6f})")
            return False

    print(f"PASS: {len(losses_run1)} steps, max diff = {max_diff:.2e}")
    return True


# ── Improvement #3: Complete Environment Locking ───────────────

def lock_environment(output_dir: str = "env_lock") -> str:
    """Capture complete environment: pip freeze, conda list, hardware info.

    Saves three files to output_dir:
        requirements_frozen.txt  — exact versions of all installed packages
        environment_info.json    — Python, PyTorch, CUDA, GPU details
        system_info.txt          — OS, kernel, driver versions

    Returns path to output_dir.
    """
    import subprocess
    import json

    os.makedirs(output_dir, exist_ok=True)

    # pip freeze — includes transitive dependencies
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            (Path(output_dir) / "requirements_frozen.txt").write_text(result.stdout)
    except Exception:
        pass

    # Environment info (PyTorch/CUDA/GPU)
    env_info = get_environment_info("pytorch")
    env_info["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    env_info["hostname"] = platform.node()
    with open(os.path.join(output_dir, "environment_info.json"), "w") as f:
        json.dump(env_info, f, indent=2, default=str)

    # System info
    sys_info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
    }
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            sys_info["nvidia_smi"] = result.stdout
    except Exception:
        pass
    with open(os.path.join(output_dir, "system_info.txt"), "w") as f:
        for k, v in sys_info.items():
            f.write(f"{k}: {v}\n")

    return output_dir


# ── Improvement #5: FSDP2 Distributed Training ─────────────────

def create_distributed_model(
    model: "nn.Module",
    strategy: str = "ddp",
    device_ids: Optional[list[int]] = None,
    **fsdp_kwargs,
) -> "nn.Module":
    """Wrap model for distributed training with DDP or FSDP2.

    Args:
        model: PyTorch model
        strategy: 'ddp', 'fsdp', or 'fsdp2'. 'ddp' uses DataParallel for single-
                  machine multi-GPU. 'fsdp'/'fsdp2' uses fully_shard (PyTorch 2.4+).
        device_ids: List of GPU device IDs. If None, uses all available GPUs.
        **fsdp_kwargs: Passed through to fully_shard (e.g., mixed_precision policy)

    Returns:
        Wrapped model

    FSDP2 is the 2025 standard for distributed training — it uses DTensor for
    sharding, has ~30% simpler API than FSDP1, and better memory management.
    """
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required")

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if strategy.lower() == "ddp":
        if len(device_ids) <= 1:
            return model.to(torch.device(f'cuda:{device_ids[0]}' if device_ids else 'cpu'))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        return model.to(torch.device(f'cuda:{device_ids[0]}'))

    elif strategy.lower() in ("fsdp", "fsdp2"):
        if _torch_version() < (2, 4):
            print(f"[FALLBACK] FSDP2 requires PyTorch >= 2.4 "
                  f"(installed: {torch.__version__}). Falling back to DDP.")
            return create_distributed_model(model, "ddp", device_ids=device_ids)
        try:
            from torch.distributed.fsdp import fully_shard
        except ImportError:
            print("[FALLBACK] torch.distributed.fsdp not available. Falling back to DDP.")
            return create_distributed_model(model, "ddp", device_ids=device_ids)
        try:
            model = model.to(torch.device(f'cuda:{device_ids[0]}'))
            model = fully_shard(model, **fsdp_kwargs)
            return model
        except (RuntimeError, ValueError) as e:
            print(f"[FALLBACK] FSDP2 init failed (distributed env not set up?): {e}")
            print("  Falling back to DDP. To use FSDP2, launch with: torchrun train.py")
            return create_distributed_model(model, "ddp", device_ids=device_ids)

    else:
        raise ValueError(f"Unknown distributed strategy: {strategy}")


# ── Version helper ──────────────────────────────────────────────

def _torch_version() -> tuple[int, int]:
    """Return (major, minor) of installed PyTorch, or (0, 0) if not found."""
    try:
        import torch
        parts = torch.__version__.split("+")[0].split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return (0, 0)


def _check_pytorch(min_ver: tuple[int, int], feature: str) -> bool:
    ok = _torch_version() >= min_ver
    if not ok:
        print(f"[SKIP] {feature} requires PyTorch >= {min_ver[0]}.{min_ver[1]} "
              f"(installed: {'.'.join(map(str, _torch_version()))})")
    return ok


# ── Improvement #6: Advanced torch.compile Patterns ────────────

def compile_model(
    model: "nn.Module",
    mode: str = "reduce-overhead",
    dynamic: bool = False,
    fullgraph: bool = False,
    use_mega_cache: bool = False,
    cache_dir: Optional[str] = None,
) -> "nn.Module":
    """Compile model with advanced PyTorch 2.6/2.7 features.

    Args:
        model: PyTorch model
        mode: 'default', 'reduce-overhead', 'max-autotune'
        dynamic: Enable dynamic shapes (for variable-length inputs)
        fullgraph: Require full-graph compilation (faster, but stricter)
        use_mega_cache: Use Mega Cache (PyTorch 2.7+) for portable cross-machine cache
        cache_dir: Directory for compilation cache artifacts

    Returns:
        Compiled model

    PyTorch 2.6+ features leveraged:
    - set_stance('eager_on_recompile'): falls back to eager on cache miss
    - Mega Cache (2.7+): save_cache_artifacts/load_cache_artifacts for cross-machine
    - Prologue Fusion: matmul preamble fused into kernel (automatic in 2.6+)
    """
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required")

    # Version gate: torch.compile requires PyTorch >= 2.0
    if _torch_version() < (2, 0):
        print(f"[SKIP] torch.compile requires PyTorch >= 2.0 "
              f"(installed: {torch.__version__}). Returning uncompiled model.")
        return model

    try:
        # Set dynamic compilation stance if supported (2.6+)
        if hasattr(torch.compiler, 'set_stance'):
            torch.compiler.set_stance("eager_on_recompile")
    except Exception:
        pass

    try:
        compiled = torch.compile(
            model,
            mode=mode,
            dynamic=dynamic,
            fullgraph=fullgraph,
        )
    except TypeError:
        compiled = torch.compile(model, mode=mode, dynamic=dynamic)
    except Exception as e:
        print(f"[SKIP] torch.compile failed: {e}. Returning uncompiled model.")
        return model

    # Mega Cache: save after compilation for reuse across machines (2.7+)
    if use_mega_cache:
        if _torch_version() >= (2, 7):
            try:
                from torch.compiler import save_cache_artifacts
                cache_path = cache_dir or ".compile_cache"
                save_cache_artifacts(compiled, cache_path)
            except Exception as e:
                print(f"[SKIP] Mega Cache save failed: {e}")
        else:
            print("[SKIP] Mega Cache requires PyTorch >= 2.7")

    return compiled


# ── Improvement #8: TorchMetrics Integration ───────────────────

def get_metrics(metric_names: list[str], num_classes: int = 0,
                 device: Optional["torch.device"] = None) -> dict:
    """Create standardized TorchMetrics for reproducible evaluation.

    TorchMetrics (2025 standard) ensures metric implementations are consistent
    across papers, avoiding subtle differences in hand-rolled metrics.

    Args:
        metric_names: List of metric names, e.g. ['Accuracy', 'F1Score', 'AUROC']
        num_classes: Number of classes for classification metrics
        device: Device to place metrics on (must match input device)

    Returns:
        Dict mapping name -> Metric instance (call .update(preds, target), .compute())
    """
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required")

    try:
        import torchmetrics
    except ImportError:
        raise ImportError("torchmetrics is required: pip install torchmetrics")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics: dict[str, object] = {}
    for name in metric_names:
        name_lower = name.lower()
        if name_lower == "accuracy":
            m = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes,
            ) if num_classes else torchmetrics.Accuracy(task="binary")
        elif name_lower == "f1score":
            m = torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes,
            ) if num_classes else torchmetrics.F1Score(task="binary")
        elif name_lower == "auroc":
            m = torchmetrics.AUROC(
                task="multiclass", num_classes=num_classes,
            ) if num_classes else torchmetrics.AUROC(task="binary")
        elif name_lower == "precision":
            m = torchmetrics.Precision(
                task="multiclass", num_classes=num_classes,
            ) if num_classes else torchmetrics.Precision(task="binary")
        elif name_lower == "recall":
            m = torchmetrics.Recall(
                task="multiclass", num_classes=num_classes,
            ) if num_classes else torchmetrics.Recall(task="binary")
        elif name_lower == "meaniou":
            m = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes)
        else:
            raise ValueError(f"Unknown metric: {name}")
        metrics[name] = m.to(device)
    return metrics


# ── Improvement #10: TorchElastic Fault-Tolerant Pattern ───────

def get_elastic_environment() -> dict:
    """Detect TorchElastic environment variables for fault-tolerant training.

    TorchElastic (torchrun) enables automatic recovery from node failures.
    Use this to conditionally configure checkpoint save/load logic.

    Returns dict with keys: is_elastic, rank, local_rank, world_size, restart_count
    """
    return {
        "is_elastic": "TORCHELASTIC_RUN_ID" in os.environ,
        "rank": int(os.environ.get("RANK", 0)),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "restart_count": int(os.environ.get("TORCHELASTIC_RESTART_COUNT", 0)),
    }


# ── Improvement #12: Safe Model Loading ────────────────────────

def safe_load_checkpoint(
    path: str,
    model: "nn.Module",
    map_location: str = "cpu",
) -> dict:
    """Securely load model checkpoint.

    PyTorch 2.6+ defaults to weights_only=True for security. This wrapper
    provides a consistent safe-loading API across PyTorch versions,
    with clear error messages for version-specific behavior.

    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        map_location: Device to map tensors to

    Returns:
        Loaded checkpoint dict (only state_dict when weights_only=True)
    """
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required")

    # PyTorch 2.6+ defaults weights_only=True for security
    try:
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        # Fallback for checkpoints with non-tensor metadata
        import warnings
        warnings.warn(
            f"Loading {path} with weights_only=False. "
            "Ensure this checkpoint is from a trusted source.",
            UserWarning,
        )
        state_dict = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    return state_dict


if __name__ == "__main__":
    print("DL Reproducibility Pack - PyTorch Patterns")
    print("=" * 50)

    print("\nGetting device...")
    device = get_device()
    print(f"Device: {device}")

    print("\nSetting seed for PyTorch...")
    set_seed(42, framework="pytorch", deterministic=True)

    print("\nEnvironment info:")
    print(get_environment_info("pytorch"))

    print("\nDeterminism check:")
    print(check_determinism("pytorch"))
