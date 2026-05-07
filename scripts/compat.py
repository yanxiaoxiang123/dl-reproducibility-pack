from __future__ import annotations

"""
compat.py - Version Compatibility Detection & Graceful Degradation

Auto-detects Python/PyTorch/OS versions and returns a feature availability
report. All skill components use this to provide graceful fallbacks.

Author: dl-reproducibility-pack v3.3
"""

import sys
from typing import Optional, Dict, Any, Tuple


def get_python_info() -> dict:
    """Detect Python version and capabilities."""
    return {
        "version": sys.version,
        "major": sys.version_info.major,
        "minor": sys.version_info.minor,
        "has_future_annotations": sys.version_info >= (3, 7),
        "has_builtin_generics": sys.version_info >= (3, 9),   # dict[str, X]
        "has_union_syntax": sys.version_info >= (3, 10),       # X | Y
    }


def get_pytorch_info() -> Optional[dict]:
    """Detect PyTorch version and capabilities. Returns None if not installed."""
    try:
        import torch
        major, minor, patch = [int(x) for x in torch.__version__.split("+")[0].split(".")[:3]]
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return {
            "version": torch.__version__,
            "major": major, "minor": minor, "patch": patch,
            "has_amp": (major, minor) >= (1, 10),
            "has_compile": (major, minor) >= (2, 0),
            "has_fsdp2": (major, minor) >= (2, 4),
            "has_mega_cache": (major, minor) >= (2, 7),
            "has_weights_only_default": (major, minor) >= (2, 6),
            "has_mps": has_mps,
            "has_cuda": has_cuda,
            "gpu_count": torch.cuda.device_count() if has_cuda else 0,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if has_cuda else [],
        }
    except ImportError:
        return None


def get_dependency_info() -> dict:
    """Check availability of optional dependencies."""
    deps = {}
    for name in ["torchmetrics", "trackio", "mlflow", "wandb", "sklearn"]:
        try:
            __import__(name)
            deps[name] = True
        except ImportError:
            deps[name] = False
    return deps


def get_os_info() -> dict:
    """Detect OS for platform-specific settings."""
    import platform
    system = platform.system()
    return {
        "system": system,
        "is_linux": system == "Linux",
        "is_macos": system == "Darwin",
        "is_windows": system == "Windows",
        "machine": platform.machine(),
        "python_64bit": sys.maxsize > 2**32,
    }


def check_compatibility(verbose: bool = True) -> dict:
    """Full compatibility check — returns report and prints warnings.

    Returns:
        dict with keys: python, pytorch, deps, os, features, warnings, recommendations
    """
    python_info = get_python_info()
    pytorch_info = get_pytorch_info()
    deps = get_dependency_info()
    os_info = get_os_info()

    features: dict[str, bool] = {}
    warnings: list[str] = []
    recommendations: list[str] = []

    # ── Python checks ──
    if python_info["major"] == 3 and python_info["minor"] < 9:
        warnings.append("Python < 3.9: use 'from typing import List, Dict' for type hints")
        recommendations.append("Upgrade to Python >= 3.9 for modern type hint syntax")

    # ── PyTorch checks ──
    if pytorch_info is None:
        warnings.append("PyTorch not installed - install: pip install torch")
        recommendations.append("Install PyTorch >= 1.10 for core reproducibility features")
    else:
        ver = (pytorch_info["major"], pytorch_info["minor"])

        features["torch_amp"] = pytorch_info["has_amp"]
        features["torch_compile"] = pytorch_info["has_compile"]
        features["fsdp2"] = pytorch_info["has_fsdp2"]
        features["mega_cache"] = pytorch_info["has_mega_cache"]
        features["weights_only_default"] = pytorch_info["has_weights_only_default"]
        features["mps_support"] = pytorch_info["has_mps"]
        features["cuda_support"] = pytorch_info["has_cuda"]

        if ver < (1, 10):
            warnings.append(f"PyTorch {pytorch_info['version']} < 1.10: torch.amp unavailable")
            recommendations.append("Upgrade to PyTorch >= 1.10")
        if ver < (2, 0):
            warnings.append(f"PyTorch {pytorch_info['version']} < 2.0: torch.compile unavailable")
            recommendations.append("Upgrade to PyTorch >= 2.0 for torch.compile")
        if ver < (2, 4):
            warnings.append(f"PyTorch {pytorch_info['version']} < 2.4: FSDP2 (fully_shard) unavailable")
        if ver < (2, 6):
            warnings.append(f"PyTorch {pytorch_info['version']} < 2.6: weights_only default is False")
        if ver < (2, 7):
            warnings.append(f"PyTorch {pytorch_info['version']} < 2.7: Mega Cache unavailable")

        if not pytorch_info["has_cuda"] and not pytorch_info["has_mps"]:
            warnings.append("No GPU detected - training will be CPU-only (slow)")

    # ── Dependency checks ──
    features["torchmetrics_available"] = deps.get("torchmetrics", False)
    features["trackio_available"] = deps.get("trackio", False)
    features["mlflow_available"] = deps.get("mlflow", False)
    features["wandb_available"] = deps.get("wandb", False)
    features["sklearn_available"] = deps.get("sklearn", False)

    if not deps.get("torchmetrics", False):
        recommendations.append("pip install torchmetrics for standardized evaluation metrics")
    if not deps.get("sklearn", False):
        recommendations.append("pip install scikit-learn for stratified dataset splits")

    # ── Print report ──
    if verbose:
        print_compat_report(python_info, pytorch_info, os_info, features, warnings, recommendations)

    return {
        "python": python_info,
        "pytorch": pytorch_info or {},
        "os": os_info,
        "deps": deps,
        "features": features,
        "warnings": warnings,
        "recommendations": recommendations,
    }


def print_compat_report(
    python_info: dict,
    pytorch_info: Optional[dict],
    os_info: dict,
    features: dict,
    warnings: list,
    recommendations: list,
) -> None:
    """Print a formatted compatibility report."""
    print("=" * 60)
    print("dl-reproducibility-pack - Compatibility Report")
    print("=" * 60)

    # Environment table
    print(f"\n{'Component':<20} {'Detected':<30} {'Status'}")
    print("-" * 60)
    py_ver = f"Python {python_info['major']}.{python_info['minor']}"
    py_ok = "OK" if python_info["major"] >= 3 and python_info["minor"] >= 9 else "OLD"
    print(f"{'Python':<20} {py_ver:<30} {py_ok}")

    if pytorch_info:
        pt_ver = f"PyTorch {pytorch_info['version']}"
        pt_ok = "OK" if (pytorch_info["major"], pytorch_info["minor"]) >= (1, 10) else "OLD"
        print(f"{'PyTorch':<20} {pt_ver:<30} {pt_ok}")
        gpu_str = f"{pytorch_info['gpu_count']} GPU(s)"
        if pytorch_info["gpu_count"] > 0:
            gpu_str += f" ({pytorch_info['gpu_names'][0]})"
        print(f"{'GPU':<20} {gpu_str:<30} {'OK' if pytorch_info['gpu_count'] > 0 else 'CPU-only'}")
        if pytorch_info["has_mps"]:
            print(f"{'MPS':<20} {'Apple Silicon':<30} OK")
    else:
        print(f"{'PyTorch':<20} {'NOT INSTALLED':<30} MISSING")

    print(f"{'OS':<20} {os_info['system']} {os_info['machine']:<30} -")
    print()

    # Features table
    if features:
        print("Feature Availability:")
        print(f"  {'Feature':<30} {'Available':<12} {'Min Version'}")
        print(f"  {'-'*28} {'-'*10} {'-'*12}")

        feature_min_ver = {
            "torch_amp": "PyTorch 1.10",
            "torch_compile": "PyTorch 2.0",
            "fsdp2": "PyTorch 2.4",
            "mega_cache": "PyTorch 2.7",
            "weights_only_default": "PyTorch 2.6",
            "mps_support": "PyTorch 1.12",
            "torchmetrics_available": "pip install",
            "sklearn_available": "pip install",
            "trackio_available": "pip install",
            "mlflow_available": "pip install",
            "wandb_available": "pip install",
        }

        for name, available in sorted(features.items()):
            min_ver = feature_min_ver.get(name, "")
            status = "OK" if available else "NO"
            print(f"  {name:<30} {status:<12} {min_ver}")
        print()

    # Warnings
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ! {w}")
        print()

    # Recommendations
    if recommendations:
        print("Recommendations:")
        for r in recommendations:
            print(f"  > {r}")
        print()

    print("=" * 60)

    # Overall verdict
    critical_warnings = sum(1 for w in warnings if "not installed" in w.lower() or "< 1.10" in w)
    if critical_warnings > 0:
        print("VERDICT: Some features will be unavailable. See warnings above.")
    else:
        print("VERDICT: All core features available. Ready for reproducible training.")
    print("=" * 60)


def feature_guard(feature_name: str, fallback_message: str = ""):
    """Decorator: skip function body if feature is unavailable, print fallback.

    Usage:
        @feature_guard("fsdp2", "FSDP2 requires PyTorch >= 2.4")
        def create_fsdp2_model(...): ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            info = get_pytorch_info()
            features = {
                "torch_compile": info and info.get("has_compile", False),
                "fsdp2": info and info.get("has_fsdp2", False),
                "mega_cache": info and info.get("has_mega_cache", False),
                "mps_support": info and info.get("has_mps", False),
            }
            if features.get(feature_name, True):
                return func(*args, **kwargs)
            else:
                msg = fallback_message or f"Feature '{feature_name}' unavailable in this environment"
                print(f"[SKIP] {msg} - {func.__name__}() not executed")
                return None
        return wrapper
    return decorator


# Run on import if script is executed directly
if __name__ == "__main__":
    check_compatibility(verbose=True)
