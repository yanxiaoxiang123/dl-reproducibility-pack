"""
dl-reproducibility-pack v2 — Deep Learning Reproducibility Toolkit

A comprehensive toolkit combining reproducibility utilities with idiomatic
PyTorch patterns for building robust, efficient, and reproducible deep
learning research projects.

v2 incorporates patterns from d2l-zh and 2025 best practices.

Author: dl-reproducibility-pack
"""

from .reproducibility import (
    set_seed,
    get_device,
    try_gpu,
    try_all_gpus,
    get_environment_info,
    check_determinism,
    log_environment_info,
    lock_environment,
    train_one_epoch,
    evaluate,
    Accumulator,
    EMA,
    create_optimizer,
    CosineWarmupScheduler,
    grad_clip,
    save_full_checkpoint,
    load_full_checkpoint,
    verify_reproducibility,
    create_distributed_model,
    compile_model,
    get_metrics,
    get_elastic_environment,
    safe_load_checkpoint,
)
from .seed_worker import (
    seed_worker,
    get_worker_seed,
    create_reproducible_dataloader,
    pad_sequence_collate,
    save_dataset_split,
    load_dataset_split,
    stratified_split,
    DatasetVersioning,
)
from .profiling import (
    ProfileContext,
    BenchmarkTimer,
    benchmark_model,
    throughput_report,
)
from .tracking import (
    ExperimentTracker,
)
from .config import Config, load_config, create_default_config

__all__ = [
    # Reproducibility
    "set_seed",
    "get_device",
    "try_gpu",
    "try_all_gpus",
    "get_environment_info",
    "check_determinism",
    "log_environment_info",
    "lock_environment",
    # Training patterns
    "train_one_epoch",
    "evaluate",
    "Accumulator",
    "EMA",
    "create_optimizer",
    "CosineWarmupScheduler",
    "grad_clip",
    # Full checkpointing (v2)
    "save_full_checkpoint",
    "load_full_checkpoint",
    "verify_reproducibility",
    # Distributed (v2)
    "create_distributed_model",
    "compile_model",
    # Metrics & hardware (v2)
    "get_metrics",
    "get_elastic_environment",
    "safe_load_checkpoint",
    # Data loading
    "seed_worker",
    "get_worker_seed",
    "create_reproducible_dataloader",
    "pad_sequence_collate",
    "save_dataset_split",
    "load_dataset_split",
    "stratified_split",
    "DatasetVersioning",
    # Profiling (v2)
    "ProfileContext",
    "BenchmarkTimer",
    "benchmark_model",
    "throughput_report",
    # Tracking (v2)
    "ExperimentTracker",
    # Configuration
    "Config",
    "load_config",
    "create_default_config",
]
