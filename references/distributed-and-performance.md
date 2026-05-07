# Distributed And Performance

Use this reference for multi-GPU, FSDP, torch.compile, profiling, or benchmark work.

## Distributed Choice

- Single GPU: keep the model plain.
- Quick single-machine multi-GPU demo: `torch.nn.DataParallel` is simple but not ideal for serious training.
- Production multi-GPU: prefer `DistributedDataParallel` launched with `torchrun`.
- Large models with memory pressure: consider FSDP/FSDP2 only when PyTorch version and distributed launch are ready.

Do not label a `DataParallel` wrapper as DDP. DDP requires process group initialization and one process per device.

## FSDP2 Notes

FSDP2-style `fully_shard` requires PyTorch 2.4+ and a distributed environment. If those are missing, fall back clearly and tell the user how to launch with `torchrun`.

## torch.compile

Use only after correctness is established.

- PyTorch 2.0+ is required.
- Dynamic shapes may trigger recompilation.
- Full-graph mode is stricter and can fail on Python side effects.
- Cache features are version-specific; treat them as optional acceleration, not a reproducibility requirement.

## Profiling

Use scheduled profiling with wait/warmup/active phases to avoid cold-start noise. For timing:

- Synchronize CUDA before and after measured blocks.
- Exclude warmup iterations.
- Report average, median, and p95 latency.
- Compute throughput as samples divided by total measured time.

## Reproducibility Tradeoffs

Performance options can conflict with exact reproducibility:

- `torch.backends.cudnn.benchmark=True` can choose nondeterministic kernels.
- TF32 changes numerical results on Ampere+ GPUs.
- AMP can change convergence and exact loss values.
- Distributed reductions may differ by topology or process count.

Document any tradeoff the patch introduces.
