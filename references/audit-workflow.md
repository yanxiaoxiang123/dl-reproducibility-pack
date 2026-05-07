# Audit Workflow

Use this reference for broad requests such as "make my DL project reproducible" or "audit this training repo".

## Discovery Checklist

- Training entrypoint and CLI arguments.
- Framework version and device assumptions.
- Config source: YAML, argparse, hydra, environment variables, notebook cells, or hardcoded constants.
- Seed calls for Python, NumPy, framework, CUDA, and hash seed.
- Dataset split creation and storage.
- DataLoader shuffle, generator, worker init, persistent workers, and custom samplers.
- Train/eval mode transitions.
- Checkpoint save/load contents.
- Resume behavior for optimizer, scheduler, AMP scaler, EMA, epoch, global step, and RNG.
- Experiment logging backend and artifact storage.
- Environment lock files and Docker/conda/pip setup.

## Diagnosis Output

For project-level work, summarize findings in this shape:

| Area | Current State | Risk | Priority | Proposed Change |
| --- | --- | --- | --- | --- |
| Seeds | Missing CUDA seed | Run drift | P0 | Add unified seed helper |

Then implement P0 first. Add P1/P2 only when the user asked for a full reproducibility pass or the project is publication-bound.

## Minimum Patch Set

For most PyTorch projects:

1. Add a seed helper and call it at process start.
2. Create stable split files and load them instead of random splitting every run.
3. Build DataLoaders with `worker_init_fn` and seeded `torch.Generator`.
4. Save full checkpoints with model, optimizer, scheduler, scaler, epoch, global step, and RNG state.
5. Restore full state on resume before continuing training.
6. Capture `pip freeze --all`, Python/PyTorch/CUDA info, GPU names, and command-line arguments.
7. Document exact training/eval/resume commands.

## Verification

Prefer a short two-run smoke test:

- Run the same small training job twice with the same seed.
- Compare loss history within an agreed tolerance.
- If exact equality is not realistic, record tolerance and known nondeterministic operations.

Do not promise bit-exact reproducibility across different GPU models, driver versions, PyTorch builds, or distributed topologies unless verified.
