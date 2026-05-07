"""
reproducibility.py - Deep Learning Reproducibility & PyTorch Patterns

A unified module combining reproducibility utilities with idiomatic PyTorch patterns
for building robust, efficient, and reproducible deep learning projects.

Usage:
    from src.reproducibility import set_seed, get_device, train_one_epoch

    device = get_device()
    set_seed(42, deterministic=True)
    loss = train_one_epoch(model, dataloader, optimizer, criterion, device)

Author: dl-reproducibility-pack
"""

import random
import os
import platform
import sys
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


def get_device() -> "torch.device":
    """
    Get the best available device (GPU preferred).

    Returns:
        torch.device: CUDA if available, otherwise CPU

    Example:
        >>> device = get_device()
        >>> print(device)
        cuda:0
    """
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# Training loop patterns following pytorch-patterns idioms
def train_one_epoch(
    model: "nn.Module",
    dataloader: "DataLoader",
    optimizer: "torch.optim.Optimizer",
    criterion: "nn.Module",
    device: "torch.device",
    scaler: Optional["torch.amp.GradScaler"] = None,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Train for one epoch following idiomatic PyTorch patterns.

    Args:
        model: The neural network model
        dataloader: Training data loader
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to train on
        scaler: Optional GradScaler for mixed precision
        max_grad_norm: Gradient clipping threshold

    Returns:
        Average loss for the epoch
    """
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required")

    model.train()
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            output = model(data)
            loss = criterion(output, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
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

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        total_loss += criterion(output, target).item()
        correct += (output.argmax(1) == target).sum().item()
        total += target.size(0)

    return total_loss / len(dataloader), correct / total


# For internal use when torch is available
def _get_tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        return None


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
