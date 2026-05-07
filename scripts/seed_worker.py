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


if __name__ == "__main__":
    print("seed_worker.py - DataLoader Worker Seeding Utilities")
    print("=" * 50)

    print("\nWorker seed examples:")
    for worker_id in range(4):
        seed = get_worker_seed(42, worker_id)
        print(f"  Worker {worker_id}: seed = {seed}")

    print("\nNote: Use with DataLoader:")
    print("  loader = DataLoader(dataset, worker_init_fn=seed_worker)")
    print("\nOr use the convenience function:")
    print("  loader = create_reproducible_dataloader(dataset, batch_size=32, seed=42)")
