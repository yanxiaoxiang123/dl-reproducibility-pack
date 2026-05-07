from __future__ import annotations

"""
seed_worker.py - DataLoader Worker Seeding Utilities

Provides worker_init_fn for PyTorch DataLoader following idiomatic patterns
to ensure reproducible data loading in multi-worker settings.

Usage:
    from src.seed_worker import seed_worker, create_reproducible_dataloader

    train_loader = create_reproducible_dataloader(
        train_dataset,
        batch_size=128,
        seed=42,
        num_workers=4
    )

Author: dl-reproducibility-pack
"""

import os
import random
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader, Dataset

# Try to import numpy - required for worker seeding
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Lazy import for torch
def _get_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def seed_worker(worker_id: int) -> None:
    """
    Seed a DataLoader worker process for reproducible data loading.

    This function should be passed as worker_init_fn to DataLoader.
    It uses torch.initial_seed() to derive a unique seed for each worker.

    Args:
        worker_id: The worker ID, automatically passed by DataLoader

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     num_workers=4,
        ...     worker_init_fn=seed_worker
        ... )
    """
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required for seed_worker")

    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for seed_worker")

    # Get the base seed from PyTorch's global seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_worker_seed(base_seed: int, worker_id: int) -> int:
    """
    Calculate the seed for a specific worker given a base seed.

    This is useful when you need to know exactly what seed a worker
    will receive for debugging or verification purposes.

    Args:
        base_seed: The main random seed (e.g., 42)
        worker_id: The DataLoader worker ID

    Returns:
        The calculated seed for that worker
    """
    return (base_seed + worker_id) % 2**32


def create_reproducible_dataloader(
    dataset: "Dataset",
    batch_size: int,
    seed: int = 42,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    collate_fn: Optional[Callable] = None,
) -> "DataLoader":
    """
    Create a fully reproducible PyTorch DataLoader following best practices.

    This is a convenience function combining:
    - Worker seeding via seed_worker
    - Generator seeding for reproducible shuffling
    - Proper pin_memory and persistent_workers settings

    Args:
        dataset: The dataset to load
        batch_size: Batch size for training
        seed: Random seed for reproducibility
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        collate_fn: Optional custom collate function

    Returns:
        DataLoader configured for reproducibility

    Example:
        >>> from torchvision.datasets import CIFAR100
        >>> from torchvision.transforms import transforms
        >>> transform = transforms.Compose([
        ...     transforms.RandomCrop(32, padding=4),
        ...     transforms.ToTensor(),
        ... ])
        >>> train_dataset = CIFAR100(root='./data', train=True, transform=transform)
        >>> train_loader = create_reproducible_dataloader(
        ...     train_dataset,
        ...     batch_size=128,
        ...     seed=42,
        ...     num_workers=4
        ... )
        >>> for batch in train_loader:
        ...     # Training loop
        ...     pass
    """
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required for create_reproducible_dataloader")

    from torch.utils.data import DataLoader

    # Create a generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)

    # persistent_workers only valid when num_workers > 0
    persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        generator=generator,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


class ReproducibleBatchSampler:
    """
    A batch sampler that ensures reproducible batch generation.

    This is useful when you need fine-grained control over batching
    or when working with custom sampling strategies.

    Args:
        sampler: The underlying sampler to use
        batch_size: Size of each batch
        drop_last: Whether to drop incomplete final batches
        seed: Random seed for reproducibility

    Example:
        >>> from torch.utils.data import RandomSampler
        >>> sampler = RandomSampler(dataset)
        >>> batch_sampler = ReproducibleBatchSampler(sampler, batch_size=32)
    """
    def __init__(
        self,
        sampler,
        batch_size: int,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed

    def __iter__(self):
        """Iterate over batches."""
        torch = _get_torch()
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        """Return the number of batches."""
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


# Efficient collate function for variable-length sequences
def pad_sequence_collate(
    batch: list[tuple],
    padding_value: float = 0.0,
    batch_first: bool = True,
) -> tuple:
    """
    Collate function that pads variable-length sequences.

    Args:
        batch: List of (sequence, label) tuples
        padding_value: Value to pad sequences with
        batch_first: If True, return (batch, seq_len), else (seq_len, batch)

    Returns:
        Tuple of (padded_sequences, labels)
    """
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required")

    sequences, labels = zip(*batch)
    padded = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=batch_first, padding_value=padding_value
    )
    return padded, torch.tensor(labels)


# ── Improvement #4: Dataset Versioning & Split Management ──────

def save_dataset_split(
    indices: dict[str, list[int]],
    output_path: str,
    metadata: Optional[dict] = None,
) -> None:
    """Save train/val/test indices to a JSON file for reproducible splits.

    Per AAAI 2025 reproducibility checklist: pre-define and save fold splits
    so every experiment uses exactly the same data partitions.

    Args:
        indices: Dict with keys 'train', 'val', 'test' → list of integer indices
        output_path: Path to save the split file (e.g. 'splits/fold_0.json')
        metadata: Optional dict with dataset name, seed, timestamp, etc.

    Example:
        >>> save_dataset_split(
        ...     {'train': train_idx, 'val': val_idx, 'test': test_idx},
        ...     'splits/fold_0.json',
        ...     metadata={'dataset': 'CIFAR100', 'seed': 42},
        ... )
    """
    import json
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    data = {"indices": indices}
    if metadata:
        data["metadata"] = metadata
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_dataset_split(path: str) -> dict[str, list[int]]:
    """Load pre-defined dataset split from JSON file.

    Returns:
        Dict with keys 'train', 'val', 'test' → list of integer indices
    """
    import json
    with open(path) as f:
        data = json.load(f)
    return data["indices"]


def stratified_split(
    labels: list[int],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Create stratified train/val/test split preserving class distribution.

    Args:
        labels: List of integer class labels for each sample
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for test
        seed: Random seed for reproducibility

    Returns:
        Dict with keys 'train', 'val', 'test' → list of integer indices
    """
    from sklearn.model_selection import train_test_split

    indices = list(range(len(labels)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=val_ratio + test_ratio,
        stratify=labels, random_state=seed,
    )

    if test_ratio > 0:
        temp_labels = [labels[i] for i in temp_idx]
        test_size_relative = test_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=test_size_relative,
            stratify=temp_labels, random_state=seed,
        )
        return {"train": train_idx, "val": val_idx, "test": test_idx}
    return {"train": train_idx, "val": temp_idx}


class DatasetVersioning:
    """Track dataset versions for reproducibility.

    >>> versioning = DatasetVersioning("cifar100", "v1.0")
    >>> versioning.record(
    ...     num_samples=50000, num_classes=100,
    ...     transforms="RandomCrop+HorizontalFlip+Normalize",
    ... )
    >>> versioning.save("data/version.json")
    """

    def __init__(self, name: str, version: str = "1.0"):
        import time as _time
        self.name = name
        self.version = version
        self.timestamp = _time.strftime("%Y-%m-%dT%H:%M:%S")
        self.metadata: dict[str, object] = {}

    def record(self, **kwargs) -> None:
        self.metadata.update(kwargs)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def save(self, path: str) -> None:
        import json
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


if __name__ == "__main__":
    print("seed_worker.py - DataLoader & Dataset Versioning Utilities")
    print("=" * 50)

    print("\nWorker seed examples:")
    for worker_id in range(4):
        seed = get_worker_seed(42, worker_id)
        print(f"  Worker {worker_id}: seed = {seed}")

    print("\nDataset versioning example:")
    ver = DatasetVersioning("cifar100", "v2")
    ver.record(num_samples=50000, num_classes=100, augmentations="RandAugment")
    print(ver.to_dict())

    print("\nNote: Use with DataLoader:")
    print("  loader = DataLoader(dataset, worker_init_fn=seed_worker)")
    print("\nOr use the convenience function:")
    print("  loader = create_reproducible_dataloader(dataset, batch_size=32, seed=42)")
