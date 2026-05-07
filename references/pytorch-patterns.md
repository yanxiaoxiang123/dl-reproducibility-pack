# PyTorch Patterns

Use this reference when editing PyTorch code.

## Seed Setup

Use one project-level seed function:

```python
def set_seed(seed: int, deterministic: bool = True) -> None:
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.allow_tf32 = False
```

Call it before model, dataset, DataLoader, or optimizer construction.

## DataLoader

Use both a worker seed function and a generator:

```python
generator = torch.Generator()
generator.manual_seed(seed)

loader = DataLoader(
    dataset,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=generator,
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
)
```

Validation and test transforms should not include random augmentation.

## Training Loop

- Call `model.train()` for training and `model.eval()` for evaluation.
- Use `optimizer.zero_grad(set_to_none=True)`.
- With AMP, call `scaler.unscale_(optimizer)` before gradient clipping.
- When using gradient accumulation, step on both accumulation boundaries and the final partial window.
- Track sample-weighted loss, especially when the last batch is smaller.

## Checkpoints

For exact resume, save:

- model state
- optimizer state
- scheduler state
- AMP scaler state
- EMA shadow state, when used
- epoch and global step
- Python RNG
- NumPy RNG
- PyTorch CPU RNG
- CUDA RNG list when CUDA is available

Use `weights_only=True` for untrusted weights. Full training checkpoints need `weights_only=False` because RNG states contain non-tensor Python objects; only load those from trusted paths.

## Metrics

Prefer TorchMetrics or a clearly documented implementation. Move metric objects to the same device as predictions. Reset state between evaluation phases.

## Common Pitfalls

- Random split computed each run.
- DataLoader has `shuffle=True` but no seeded generator.
- Worker processes use NumPy/random without worker seeding.
- Resume restores model weights but not optimizer/scheduler/RNG.
- Validation loss is averaged by batches instead of samples.
- Dropout or BatchNorm left in train mode during evaluation.
- TF32 enabled when exact reproducibility is required.
