# dl-reproducibility-pack v3.1.2

[![version](https://img.shields.io/badge/version-v3.1.2-blue)](https://github.com/yanxiaoxiang123/dl-reproducibility-pack/releases/tag/v3.1.2)
[![license](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![tested](https://img.shields.io/badge/tested-RTX%203090%20%C3%972%20%7C%20PyTorch%202.9.1%20%7C%20CUDA%2012.8-brightgreen)]()

Deep learning reproducibility toolkit for PyTorch and TensorFlow researchers.  
**v3.1.2** — 实地测试通过：双 RTX 3090，PyTorch 2.9.1，CUDA 12.8，20/20 测试全部通过。

---

## v3.1.2 Bug 修复 (2026-05-07)

在 `newsstock` conda 环境中实地测试发现的 4 个运行时问题已修复：

| Bug | 现象 | 根因 | 修复 |
|-----|------|------|------|
| **#1** `EMA` forward reference | `NameError: name 'EMA' is not defined` | `train_one_epoch()` 的 `ema: Optional[EMA]` 类型注解在 `EMA` 类定义之前被求值 | 添加 `from __future__ import annotations` |
| **#2** `torch` 模块级装饰器 | `NameError: name 'torch' is not defined` | `@torch.no_grad()` 装饰器在 lazy import (`_get_torch()`) 可用前被执行 | 改为函数内部 `with torch.no_grad():` 上下文管理器 |
| **#3** `seed_worker.py` 缺少导入 | `NameError: name 'os' is not defined` | v3 新增的 `save_dataset_split()` / `DatasetVersioning` 使用了 `os` 但未导入 | 添加 `import os` |
| **#4** TorchMetrics 设备不匹配 | `RuntimeError: Expected all tensors to be on the same device` | `get_metrics()` 创建的 metric 对象默认在 CPU，输入在 GPU | 添加 `device` 参数并调 `.to(device)` |

修复后 20/20 测试全部通过，包括 `verify_reproducibility()` 确认两次运行的 loss 曲线逐位一致（max diff = 0.00e+00）。

---

## 版本演化：v1 → v2 → v3

```
v1 (基础)          v2 (d2l-zh 训练设施)        v3 (2025 可复现性标准)
─────────         ──────────────────          ──────────────────────────
种子设定 ✓         Accumulator 指标追踪        完整 RNG 检查点 ★
设备无关 ✓         EMA 权重平均                二跑一致性验证 ★
训练循环 ✓         CosineWarmup 调度器         环境完整锁定 ★
数据加载 ✓         优化器工厂                   数据集版本控制 ★
检查点 ✓           梯度管理                     FSDP2 分布式训练 ★
环境 ✓             权重初始化目录              高级 torch.compile ★
项目结构 ✓         多 GPU (DP/DDP)             Profiling & 性能 ★
配置管理 ✓         数据增强                     TorchMetrics 标准化 ★
文档模板 ✓         调试 & 数值稳定性           实验追踪 (Trackio/MLflow/W&B) ★
                   快速参考 22 项              TorchElastic 容错 ★
                   配置字段 20 项              多平台 (CUDA/MPS/CPU) ★
                                              安全加载 ★
                                              快速参考 35 项
                                              配置字段 30+ 项
                                              8 个脚本文件
```

### v1 → v2 关键变化（10 项）

| 领域 | v1 | v2 新增 |
|------|----|----------|
| 指标追踪 | 原始 `float` 变量 | `Accumulator` 类 |
| 学习率调度 | 仅配置，无代码 | `CosineWarmupScheduler` |
| 权重平均 | 不支持 | `EMA` 类 |
| 优化器设置 | 手动调用 | `create_optimizer()` 工厂 |
| 梯度裁剪 | 硬编码 | `grad_clip()` + AMP unscale 流程 |
| 梯度累积 | 不支持 | 内置于 `train_one_epoch()` |
| 权重初始化 | 仅 Kaiming | 完整目录：Kaiming/Xavier/Normal |
| 多 GPU | 不支持 | DataParallel + DDP |
| 数据增强 | 不支持 | Train/val 分离、RandAugment |
| 调试 | 不支持 | 异常检测、NaN 检查、早停 |

### v2 → v3 关键变化（12 项）

| # | 新增 | 关键函数/类 |
|---|------|------------|
| 1 | 完整 RNG 检查点 | `save_full_checkpoint()` / `load_full_checkpoint()` |
| 2 | 二跑验证 | `verify_reproducibility()` |
| 3 | 环境锁定 | `lock_environment()` |
| 4 | 数据集版本控制 | `save/load_dataset_split()` / `DatasetVersioning` |
| 5 | FSDP2 分布式训练 | `create_distributed_model(strategy="fsdp2")` |
| 6 | 高级 torch.compile | `compile_model(use_mega_cache=True)` |
| 7 | 性能分析 | `ProfileContext` / `BenchmarkTimer` / `benchmark_model()` |
| 8 | 标准化评估 | `get_metrics(["Accuracy","F1Score","AUROC"])` |
| 9 | 实验追踪 | `ExperimentTracker("trackio"/"mlflow"/"wandb")` |
| 10 | 容错训练 | `get_elastic_environment()` |
| 11 | 多平台设备 | `get_device()` → CUDA / MPS / CPU |
| 12 | 安全加载 | `safe_load_checkpoint()` |

---

## Interactive Diagnostic Workflow (Plan-First)

When you say "帮我把项目改成可复现的", the skill does NOT immediately generate code.
Instead, it follows a **诊断→计划→实施** (Diagnose→Plan→Implement) workflow:

### Phase 1 — Diagnostic Questionnaire
Claude asks 12 targeted questions across 4 categories:
- **A. 项目基础**: Framework, domain, hardware, code maturity
- **B. 可复现性需求**: Publication target, bit-exact vs tolerance, resume needs, seed count
- **C. 数据与基础设施**: Dataset situation, experiment tracking, Docker, Python env
- **D. 代码检查**: Auto-scans for existing seeds, train/eval patterns, checkpoint code

### Phase 2 — Diagnosis & Tailored Plan
Presents a diagnosis table + prioritized implementation plan:
- **P0 (Must):** Seeds, environment lock, data splits
- **P1 (Strongly Recommended):** Gradient mgmt, LR schedule, checkpoint, Docker
- **P2 (Nice to Have):** Tracking, profiling, verification CI
- **P3 (Optional):** AMP, FSDP2, torch.compile, TorchElastic

Priority adjusts based on publication target, hardware, domain, and framework.

### Phase 3 — Implementation
After user approves the plan, implements each step with diff previews.

---

## Overview

This skill helps researchers make their deep learning projects fully reproducible:

- **Interactive diagnostic** — asks the right questions before recommending solutions
- **AAAI 2025 Reproducibility Checklist** compliance
- **Full RNG checkpointing** — save ALL random states (Python/NumPy/PyTorch/CUDA)
- **Two-run verification** — CI/CD guard for bit-exact reproducibility
- **Environment locking** — pip freeze + nvidia-smi + hardware metadata
- **Dataset versioning** — pre-defined splits, stratified sampling, version tracking
- **FSDP2** — distributed training with PyTorch 2.4+ fully_shard
- **Advanced torch.compile** — Mega Cache (2.7+), Prologue Fusion, set_stance
- **Profiling + Benchmarks** — PyTorch Profiler with warmup/active/wait scheduling
- **TorchMetrics** — standardized, peer-reviewed metric implementations
- **Experiment tracking** — unified Trackio / MLflow / W&B interface
- **TorchElastic** — fault-tolerant training with auto node recovery
- **Multi-platform** — CUDA + Apple MPS + CPU automatic detection

---

## Installation

### Option 1: Install via Claude Code

```
/plugin marketplace add <this-skill>
```

### Option 2: Manual Installation

```bash
# Global installation
~/.claude/skills/dl-reproducibility-pack/

# Or project-local
your-project/.claude/skills/dl-reproducibility-pack/
```

---

## Components

### SKILL.md (~1400 lines)

31 topics across 12 major sections — model architecture, weight initialization,
training & evaluation, metric tracking, gradient management, LR scheduling,
EMA, multi-GPU & FSDP2, debugging & diagnostics, data pipelines, augmentation,
checkpointing, environment locking, profiling, tracking, and documentation templates.

### scripts/ (8 files)

| 文件 | 职责 |
|------|------|
| `reproducibility.py` | Seeds, device, Accumulator, EMA, CosineWarmup, full RNG checkpoint, FSDP2, torch.compile, TorchMetrics, TorchElastic, safe loading |
| `seed_worker.py` | DataLoader seeding, reproducible loader factory, dataset split management, versioning |
| `profiling.py` | **NEW v3** — ProfileContext, BenchmarkTimer, benchmark_model, throughput_report |
| `tracking.py` | **NEW v3** — ExperimentTracker (Trackio/MLflow/W&B) |
| `config.py` | 6 dataclass sections, 30+ fields, YAML I/O, dot-path access |

---

## Example: Full Training Pipeline (v3)

```python
from src.reproducibility import (
    set_seed, get_device,
    Accumulator, EMA, CosineWarmupScheduler, create_optimizer,
    train_one_epoch, evaluate,
    save_full_checkpoint, verify_reproducibility, lock_environment,
    compile_model, get_metrics,
)
from src.seed_worker import create_reproducible_dataloader, save_dataset_split, stratified_split
from src.tracking import ExperimentTracker

# ── 1. Lock environment for paper reproducibility ──
lock_environment("env_lock")

# ── 2. Determinism ──
device = get_device()  # cuda > mps > cpu
set_seed(42)

# ── 3. Dataset with versioned splits ──
split = stratified_split(all_labels, val_ratio=0.1, test_ratio=0.1, seed=42)
save_dataset_split(split, "splits/fold_0.json",
                   metadata={"dataset": "CIFAR100", "seed": 42})

train_loader = create_reproducible_dataloader(train_dataset, batch_size=128, seed=42)
val_loader   = create_reproducible_dataloader(val_dataset,   batch_size=256, seed=42, shuffle=False)

# ── 4. Model with advanced compilation ──
model = MyModel().to(device)
model = compile_model(model, mode="reduce-overhead", use_mega_cache=True)

# ── 5. Training infrastructure ──
optimizer = create_optimizer(model, "AdamW", lr=0.001, weight_decay=0.0001)
scheduler = CosineWarmupScheduler(optimizer, warmup_steps=500, total_steps=10000)
ema = EMA(model, decay=0.999)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.amp.GradScaler("cuda")

# ── 6. TorchMetrics for standardized eval ──
metrics = get_metrics(["Accuracy", "F1Score", "AUROC"], num_classes=100)

# ── 7. Experiment tracking ──
tracker = ExperimentTracker("trackio", "resnet50_cifar100")
tracker.log_hyperparams({"lr": 0.001, "batch_size": 128, "seed": 42})

# ── 8. Training loop ──
for epoch in range(num_epochs):
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device,
        scaler=scaler, max_grad_norm=1.0, accumulation_steps=4, ema=ema,
    )
    scheduler.step()

    # Eval with EMA weights + TorchMetrics
    ema.apply_shadow()
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    ema.restore()

    tracker.log_metrics({"train_loss": train_loss, "val_acc": val_acc}, step=epoch)
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

# ── 9. Full RNG checkpoint for exact resume ──
save_full_checkpoint(
    model, optimizer, "checkpoint_final.pt",
    epoch=num_epochs, global_step=num_epochs * len(train_loader),
    loss=train_loss, scheduler=scheduler, ema=ema, scaler=scaler,
)

tracker.finish()
```

---

## v3 新增导入一览

```python
from src.reproducibility import (
    # v3 新增 (12 项改进)
    save_full_checkpoint,      # #1: 完整 RNG 检查点
    load_full_checkpoint,
    verify_reproducibility,    # #2: 二跑验证
    lock_environment,          # #3: 环境锁定
    create_distributed_model,  # #5: FSDP2
    compile_model,             # #6: 高级 torch.compile
    get_metrics,               # #8: TorchMetrics
    get_elastic_environment,   # #10: TorchElastic
    safe_load_checkpoint,      # #12: 安全加载
    try_gpu, try_all_gpus,     # #11: 多平台设备

    # v2 已有
    Accumulator, EMA, create_optimizer, CosineWarmupScheduler, grad_clip,
    train_one_epoch, evaluate, set_seed, get_device,
)
from src.seed_worker import (
    # v3 新增 (#4: 数据集版本控制)
    save_dataset_split, load_dataset_split, stratified_split, DatasetVersioning,

    # v2 已有
    seed_worker, create_reproducible_dataloader, pad_sequence_collate,
)
from src.profiling import ProfileContext, BenchmarkTimer, benchmark_model   # v3 新增 (#7)
from src.tracking import ExperimentTracker                                   # v3 新增 (#9)
```

---

## License

MIT License

---

## Contributing

Contributions welcome! Please submit issues and pull requests on GitHub.
