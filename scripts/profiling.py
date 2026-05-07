"""
profiling.py - Training Performance Profiling Utilities

Provides PyTorch Profiler wrappers for benchmarking and diagnosing
training bottlenecks. Based on PyTorch 2.6+ best practices.

Author: dl-reproducibility-pack v2
"""

import os
import time
from typing import Optional, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader
    from torch import nn


def _get_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


class ProfileContext:
    """Context manager for PyTorch Profiler with schedule-based profiling.

    Uses warmup → active → wait pattern to avoid profiling overhead on
    early iterations (cold cache, JIT warmup).

    >>> with ProfileContext(model, log_dir="./profiler_logs", active=3) as prof:
    ...     for batch in train_loader:
    ...         loss.backward()
    ...         optimizer.step()
    ...         prof.step()

    Open results with: tensorboard --logdir=./profiler_logs
    Or use Chrome Trace: open profiler_logs/*.chrome_trace in chrome://tracing
    """

    def __init__(
        self,
        log_dir: str = "./profiler_logs",
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True,
        with_flops: bool = False,
    ):
        torch = _get_torch()
        if torch is None:
            raise ImportError("torch is required")

        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        schedule = torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=1,
        )
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]

        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
        )

    def __enter__(self):
        self.profiler.__enter__()
        return self.profiler

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.__exit__(exc_type, exc_val, exc_tb)


class BenchmarkTimer:
    """Lightweight CUDA-synchronized timer for micro-benchmarking.

    >>> timer = BenchmarkTimer()
    >>> with timer:
    ...     output = model(data)
    >>> print(f"Forward pass: {timer.elapsed_ms:.2f}ms")
    """

    def __init__(self, cuda_sync: bool = True):
        self.cuda_sync = cuda_sync
        self._start: float = 0.0
        self._end: float = 0.0
        self.history: list[float] = []

    def __enter__(self):
        torch = _get_torch()
        if torch and self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        torch = _get_torch()
        if torch and self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._end = time.perf_counter()
        self.history.append(self.elapsed_ms)

    @property
    def elapsed_ms(self) -> float:
        return (self._end - self._start) * 1000

    @property
    def avg_ms(self) -> float:
        return sum(self.history) / len(self.history) if self.history else 0.0

    @property
    def median_ms(self) -> float:
        if not self.history:
            return 0.0
        sorted_times = sorted(self.history)
        n = len(sorted_times)
        if n % 2 == 0:
            return (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        return sorted_times[n // 2]


def benchmark_model(
    model: "nn.Module",
    input_shape: tuple[int, ...],
    device: "torch.device",
    num_warmup: int = 10,
    num_runs: int = 100,
    mode: str = "train",
) -> dict:
    """Benchmark model throughput and memory usage.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape including batch dim, e.g. (128, 3, 224, 224)
        device: Device to run on
        num_warmup: Warmup iterations (excluded from stats)
        num_runs: Measured iterations
        mode: 'train' (includes backward) or 'inference' (forward only)

    Returns:
        Dict with keys: throughput_imgs_per_sec, latency_ms, gpu_memory_mb
    """
    torch = _get_torch()
    if torch is None:
        raise ImportError("torch is required")

    model = model.to(device)
    model.train() if mode == "train" else model.eval()
    input_tensor = torch.randn(*input_shape, device=device)

    # Warmup
    for _ in range(num_warmup):
        if mode == "train":
            output = model(input_tensor)
            loss = output.sum()
            loss.backward()
        else:
            with torch.no_grad():
                model(input_tensor)

    # Measure
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    timer = BenchmarkTimer()
    for _ in range(num_runs):
        with timer:
            if mode == "train":
                output = model(input_tensor)
                loss = output.sum()
                loss.backward()
            else:
                with torch.no_grad():
                    model(input_tensor)

    batch_size = input_shape[0]
    throughput = (batch_size * num_runs) / (timer.avg_ms / 1000) if timer.history else 0.0

    result = {
        "batch_size": batch_size,
        "num_runs": num_runs,
        "mode": mode,
        "throughput_imgs_per_sec": throughput,
        "latency_ms_avg": timer.avg_ms,
        "latency_ms_median": timer.median_ms,
        "latency_ms_p95": 0.0,
    }

    # P95 latency
    if len(timer.history) >= 20:
        sorted_times = sorted(timer.history)
        result["latency_ms_p95"] = sorted_times[int(len(sorted_times) * 0.95)]

    # GPU memory
    if torch.cuda.is_available():
        result["gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        result["gpu_memory_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    return result


def throughput_report(model: "nn.Module", device: "torch.device", **kwargs) -> None:
    """Print a formatted throughput report."""
    result = benchmark_model(model, device=device, **kwargs)
    print("=" * 60)
    print(f"Benchmark: {result['mode'].upper()} mode")
    print(f"  Batch size:    {result['batch_size']}")
    print(f"  Throughput:    {result['throughput_imgs_per_sec']:.1f} imgs/sec")
    print(f"  Latency (avg): {result['latency_ms_avg']:.3f} ms")
    print(f"  Latency (p95): {result['latency_ms_p95']:.3f} ms")
    if "gpu_memory_mb" in result:
        print(f"  GPU Memory:    {result['gpu_memory_mb']:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    print("Profiling utilities for dl-reproducibility-pack v2")
    print("  ProfileContext  — PyTorch Profiler with warmup/active/wait scheduling")
    print("  BenchmarkTimer  — CUDA-synchronized micro-benchmark timer")
    print("  benchmark_model — Throughput & memory benchmark")
