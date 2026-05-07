---
name: dl-reproducibility-pack
description: Deep learning reproducibility workflow and utility pack for PyTorch or TensorFlow research projects. Use when the user wants to make training code reproducible, audit determinism, scaffold a research project, lock an environment, create reproducible dataset splits, improve checkpoints, set seeds, diagnose DataLoader randomness, add experiment tracking, benchmark/profile training, prepare publication reproducibility materials, or modernize DL training loops for CPU/CUDA/MPS, AMP, DDP/FSDP-style distributed training, torch.compile, or full RNG checkpoint resume.
---

# Deep Learning Reproducibility Pack

Use this skill to turn a deep learning project into a reproducible research artifact. Prefer diagnosing the existing project first, then make the smallest set of changes that give the user reliable reruns, resumable checkpoints, clear environment records, and publishable documentation.

## First Move

1. Identify the framework, training entrypoint, dataset construction, checkpoint format, and environment manager.
2. Run `python scripts/compat.py` from this skill folder when compatibility is unclear.
3. Inspect the target project for existing seed handling, DataLoader construction, train/eval mode usage, checkpointing, config files, and logging.
4. Present a short prioritized plan for broad project rewrites. For narrow requests, implement directly.

## Priority Model

- P0: seeds, deterministic flags, worker seeding, fixed dataset splits, device-safe code, train/eval mode correctness.
- P1: full checkpoints with optimizer/scheduler/scaler/RNG states, environment lock, config file, weighted evaluation metrics, resume verification.
- P2: experiment tracking, profiling/benchmarking, reproducibility README, citation metadata, CI two-run smoke test.
- P3: AMP, torch.compile, DDP/FSDP2, TorchElastic, advanced metric collections. Add these only when the project and hardware justify them.

## Bundled Scripts

Use scripts directly where possible instead of rewriting common infrastructure.

| File | Use For |
| --- | --- |
| `scripts/compat.py` | Python/PyTorch/CUDA/dependency compatibility report and feature gates. |
| `scripts/reproducibility.py` | Seeding, device selection, train/eval helpers, EMA, optimizer/scheduler helpers, full RNG checkpoints, environment lock, compile helpers, metrics, safe loading. |
| `scripts/seed_worker.py` | DataLoader worker seeding, reproducible DataLoader factory, dataset split save/load, stratified split, dataset version metadata. |
| `scripts/config.py` | Dataclass-backed YAML config with update/get helpers. |
| `scripts/profiling.py` | PyTorch profiler wrapper, CUDA-synchronized timer, throughput and latency benchmark. |
| `scripts/tracking.py` | Unified dummy/Trackio/MLflow/W&B experiment tracker wrapper. |

## When To Read References

- Read `references/audit-workflow.md` for project-level audits or requests like "make this repo reproducible".
- Read `references/pytorch-patterns.md` when modifying training loops, DataLoaders, checkpoints, metrics, AMP, or device handling.
- Read `references/distributed-and-performance.md` for multi-GPU, FSDP2, torch.compile, profiling, or benchmarking requests.
- Read `references/project-artifacts.md` when generating README, CITATION, environment files, project structure, or publication checklists.

## Implementation Rules

- Preserve the user's existing project style and framework choices.
- Make determinism explicit but explain speed tradeoffs when disabling TF32, CuDNN benchmark, or non-deterministic algorithms.
- Save and restore Python, NumPy, PyTorch CPU, and CUDA RNG states for exact resume.
- Seed DataLoader workers and pass a seeded `torch.Generator` when shuffling.
- Store dataset split indices in files instead of recomputing random splits at runtime.
- Keep validation/test transforms deterministic and separate from training augmentation.
- Weight evaluation loss by sample count, not by number of batches.
- For gradient accumulation, step the optimizer at the final partial accumulation window.
- Prefer `weights_only=True` for untrusted model weights. Use `weights_only=False` only for trusted full checkpoints that contain RNG or Python objects.
- Do not add distributed training, torch.compile, or tracking backends unless the project can run them or the user asks for them.

## Common Task Routes

### Make A Project Reproducible

1. Read `references/audit-workflow.md`.
2. Inspect project files and identify the training entrypoint.
3. Add or adapt `set_seed`, `seed_worker`, fixed split files, and config seed fields.
4. Upgrade checkpoints to include optimizer, scheduler, scaler, epoch/step, and RNG states.
5. Add `lock_environment()` or equivalent environment capture.
6. Add a two-run smoke verification when runtime is reasonable.
7. Update README with exact commands and reproducibility notes.

### Fix Non-Reproducible Training

1. Check whether data splits, shuffling, augmentations, and worker seeds are stable.
2. Check CuDNN benchmark, deterministic algorithms, TF32, AMP, dropout/train/eval mode, and checkpoint resume state.
3. Compare two short runs with the same seed and record the first divergent metric, batch, or RNG-dependent operation.
4. Patch the smallest source of nondeterminism first.

### Scaffold Reproducible Research Files

1. Read `references/project-artifacts.md`.
2. Create config, split, checkpoint, environment, and documentation files that match the user's existing project layout.
3. Include exact commands for training, evaluation, resume, and environment locking.

## Compatibility Notes

- Core helpers target Python 3.8+ and PyTorch 1.10+.
- `torch.compile` requires PyTorch 2.0+ and should fall back cleanly.
- FSDP2-style `fully_shard` requires PyTorch 2.4+ and a distributed launch context.
- PyTorch 2.6+ changed safe-loading defaults around `weights_only`.
- Some deterministic settings reduce throughput; document that tradeoff.
