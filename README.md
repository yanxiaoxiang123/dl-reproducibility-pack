# dl-reproducibility-skills v3.3

[![version](https://img.shields.io/badge/version-v3.3.0-blue)](https://github.com/yanxiaoxiang123/dl-reproducibility-skills/releases/tag/v3.3.0)
[![license](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![tested](https://img.shields.io/badge/tested-RTX%203090%20%C3%972%20%7C%20PyTorch%202.9.1%20%7C%20CUDA%2012.8-brightgreen)]()
[![python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![pytorch](https://img.shields.io/badge/pytorch-1.10%2B-orange)]()

Deep learning reproducibility toolkit for PyTorch and TensorFlow researchers.  
**v3.3** — navigation-first skill 架构，按需 references，修复梯度累积、评估 loss、benchmark 吞吐量和 Windows/Python 3.8 兼容问题。

---

## v3.3 新增：Skill 瘦身、按需引用与训练可靠性修复 (2026-05-07)

v3.3 把 `SKILL.md` 从长教程改成轻量导航文件，详细知识拆到 `references/` 按需加载，减少上下文占用并提升 agent 执行稳定性。

**结构优化：**

| 变更 | 说明 |
|------|------|
| `SKILL.md` navigation-first | 仅保留触发意图、优先级、脚本索引、执行规则和常见任务路线 |
| `references/audit-workflow.md` | 项目级可复现性审计和改造流程 |
| `references/pytorch-patterns.md` | PyTorch seeds、DataLoader、checkpoint、metrics、训练循环模式 |
| `references/distributed-and-performance.md` | DDP/FSDP2、torch.compile、profiling、benchmark 指南 |
| `references/project-artifacts.md` | README、CITATION、环境锁定、项目结构模板 |

**可靠性修复：**

| 文件 | 修复 |
|------|------|
| `reproducibility.py` | 梯度累积最后不足整窗时仍执行 optimizer step，并按实际窗口缩放 loss |
| `reproducibility.py` | `evaluate()` 改为按样本数加权 validation loss |
| `profiling.py` | 修正 throughput 高估 `num_runs` 倍的问题，并修正 p95 索引 |
| `seed_worker.py` / `profiling.py` | 添加 postponed annotations，兼容 Python 3.8 |
| `compat.py` | 输出改为 ASCII，避免 Windows GBK 控制台 UnicodeEncodeError |

验证：`compileall`、逐文件导入、`scripts/compat.py`、skill validator、梯度累积烟测均通过。

---

## v3.2 新增：版本兼容性 & 优雅降级 (2026-05-07)

用户 Python/PyTorch 版本千差万别，v3.2 解决"换个环境就崩溃"的问题。

**`compat.py` — 一键环境诊断：**

```bash
python scripts/compat.py
```
输出：
```
============================================================
dl-reproducibility-skills - Compatibility Report
============================================================
Component            Detected                       Status
------------------------------------------------------------
Python               Python 3.10                    OK
PyTorch              PyTorch 2.9.1+cu128            OK
GPU                  2 GPU(s) (NVIDIA GeForce RTX 3090) OK
...
Feature Availability:
  torch_compile                  OK           PyTorch 2.0
  fsdp2                          OK           PyTorch 2.4
  mega_cache                     OK           PyTorch 2.7
  mps_support                    NO           PyTorch 1.12 (not on Linux)
============================================================
VERDICT: All core features available.
```

**所有版本敏感函数添加防护门，不会因版本不匹配崩溃：**

| 函数 | 最低 PT 版本 | 不满足时的行为 |
|------|-------------|---------------|
| `compile_model()` | 2.0 | 返回未编译模型 + `[SKIP] torch.compile requires PyTorch >= 2.0` |
| `compile_model(use_mega_cache=True)` | 2.7 | 跳过 Mega Cache + `[SKIP]` 提示 |
| `create_distributed_model("fsdp2")` | 2.4 | 自动回退到 DDP + `[FALLBACK]` 提示 |
| `create_distributed_model("fsdp2")` | 分布式环境未初始化 | 自动回退到 DDP + 提示 `torchrun train.py` |
| `get_device(allow_mps=True)` | 1.12 | MPS 跳过，回退到 CPU |
| `get_metrics()` | torchmetrics 未安装 | `ImportError: pip install torchmetrics` |
| `stratified_split()` | sklearn 未安装 | `ImportError: pip install scikit-learn` |

**核心原则**：能跑的功能全跑，不能跑的跳过并告知用户，绝不崩溃。

---

## 版本兼容性矩阵

| 你的 PyTorch → | 1.10−1.13 | 2.0−2.3 | 2.4+ |
|---------------|-----------|---------|------|
| 核心功能 (seeds, EMA, scheduler, 检查点) | ✓ | ✓ | ✓ |
| AMP 混合精度 | ✓ | ✓ | ✓ |
| `torch.compile` | 跳过 | ✓ | ✓ |
| FSDP2 | 回退到 DDP | 回退到 DDP | ✓ |
| Mega Cache | 跳过 | 跳过 | ✓ (2.7+) |
| MPS (Apple Silicon) | ✓ (1.12+) | ✓ | ✓ |

---

<details>
<summary>📋 v3.1.2 Bug 修复 (2026-05-07)</summary>

| Bug | 现象 | 修复 |
|-----|------|------|
| `EMA` forward reference | `NameError` | 添加 `from __future__ import annotations` |
| `@torch.no_grad()` 装饰器 | `NameError` | 改为 `with torch.no_grad():` |
| `seed_worker.py` 缺少 `import os` | `NameError` | 添加导入 |
| TorchMetrics 设备不匹配 | `RuntimeError` | `get_metrics()` 加 `device` 参数 |

</details>

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
                                              6 个脚本文件 + references
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
~/.claude/skills/dl-reproducibility-skills/

# Or project-local
your-project/.claude/skills/dl-reproducibility-skills/
```

---

## Components

### SKILL.md (navigation-first)

Concise agent instructions for diagnosis, prioritization, bundled script usage,
and when to load deeper references. Long examples live in `references/` so the
skill does not overload the context window on every activation.

### scripts/ (6 files)

| 文件 | 职责 |
|------|------|
| `compat.py` | 版本检测：`check_compatibility()` 一键诊断环境 |
| `reproducibility.py` | Seeds, device, EMA, CosineWarmup, full RNG checkpoint, FSDP2, torch.compile, TorchMetrics, TorchElastic, safe loading |
| `seed_worker.py` | DataLoader seeding, reproducible loader factory, dataset split management, versioning |
| `profiling.py` | ProfileContext, BenchmarkTimer, benchmark_model, throughput_report |
| `tracking.py` | ExperimentTracker (Trackio/MLflow/W&B) |
| `config.py` | 6 dataclass sections, 30+ fields, YAML I/O, dot-path access |

### references/ (4 files)

| 文件 | 何时读取 |
|------|----------|
| `audit-workflow.md` | 项目级可复现性审计和改造 |
| `pytorch-patterns.md` | PyTorch 训练循环、DataLoader、checkpoint、metrics |
| `distributed-and-performance.md` | DDP/FSDP2、torch.compile、profiling、benchmark |
| `project-artifacts.md` | README、CITATION、环境锁定、项目结构 |

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
from src.compat import check_compatibility                                   # 版本诊断
```

---

## License

MIT License

---

## Contributing

Contributions welcome! Please submit issues and pull requests on GitHub.
