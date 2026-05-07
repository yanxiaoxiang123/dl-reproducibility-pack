"""
dl-reproducibility-pack - Deep Learning Reproducibility Toolkit

A comprehensive toolkit combining reproducibility utilities with idiomatic
PyTorch patterns for building robust, efficient, and reproducible deep
learning research projects.

Author: dl-reproducibility-pack
"""

from .reproducibility import (
    set_seed,
    get_device,
    get_environment_info,
    check_determinism,
    log_environment_info,
    train_one_epoch,
    evaluate,
)
from .seed_worker import (
    seed_worker,
    get_worker_seed,
    create_reproducible_dataloader,
    pad_sequence_collate,
)
from .config import Config, load_config, create_default_config

__all__ = [
    # Reproducibility
    "set_seed",
    "get_device",
    "get_environment_info",
    "check_determinism",
    "log_environment_info",
    # Training patterns
    "train_one_epoch",
    "evaluate",
    # Data loading
    "seed_worker",
    "get_worker_seed",
    "create_reproducible_dataloader",
    "pad_sequence_collate",
    # Configuration
    "Config",
    "load_config",
    "create_default_config",
]
