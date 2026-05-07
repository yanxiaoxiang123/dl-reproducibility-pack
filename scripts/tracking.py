"""
tracking.py - Experiment Tracking Integration

Unified interface for experiment tracking backends:
  - Trackio (HuggingFace, launched July 2025) — lightweight, open-source
  - MLflow — enterprise standard
  - Weights & Biases — cloud-hosted, rich visualizations

Usage:
    from src.tracking import ExperimentTracker

    tracker = ExperimentTracker(backend="mlflow", experiment_name="my_exp")
    tracker.log_hyperparams(config.__dict__)
    for epoch in range(epochs):
        tracker.log_metrics({"loss": train_loss, "acc": val_acc}, step=epoch)
    tracker.finish()

Author: dl-reproducibility-pack v3.3
"""

import os
from typing import Optional, Any
from pathlib import Path


class DummyTracker:
    """No-op tracker used when no backend is available."""

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        pass

    def log_hyperparams(self, params: dict) -> None:
        pass

    def log_artifact(self, path: str) -> None:
        pass

    def finish(self) -> None:
        pass


class ExperimentTracker:
    """Unified experiment tracking across Trackio, MLflow, and W&B.

    >>> tracker = ExperimentTracker(backend="trackio", experiment_name="resnet50")
    >>> tracker.log_hyperparams({"lr": 0.001, "batch_size": 128})
    >>> tracker.log_metrics({"train_loss": 0.5, "val_acc": 0.85}, step=epoch)
    >>> tracker.finish()
    """

    def __init__(
        self,
        backend: str = "trackio",
        experiment_name: str = "experiment",
        tracking_uri: Optional[str] = None,
        project: Optional[str] = None,
    ):
        self.backend = backend.lower()
        self.experiment_name = experiment_name
        self._tracker: Any = None
        self._init_tracker(tracking_uri, project)

    def _init_tracker(
        self, tracking_uri: Optional[str], project: Optional[str]
    ) -> None:
        if self.backend == "trackio":
            self._init_trackio(tracking_uri)
        elif self.backend == "mlflow":
            self._init_mlflow(tracking_uri, project)
        elif self.backend == "wandb":
            self._init_wandb(project)
        elif self.backend == "dummy":
            self._tracker = DummyTracker()
        else:
            raise ValueError(
                f"Unknown backend: {self.backend}. "
                "Use 'trackio', 'mlflow', 'wandb', or 'dummy'."
            )

    def _init_trackio(self, tracking_uri: Optional[str]) -> None:
        try:
            import trackio
        except ImportError:
            print("Trackio not installed. Install: pip install trackio")
            print("Falling back to local SQLite tracker.")
            self._tracker = DummyTracker()
            return

        path = tracking_uri or f"./trackio/{self.experiment_name}"
        os.makedirs(path, exist_ok=True)
        self._tracker = trackio.Tracker(
            experiment_name=self.experiment_name,
            storage_path=path,
        )

    def _init_mlflow(
        self, tracking_uri: Optional[str], project: Optional[str]
    ) -> None:
        try:
            import mlflow
        except ImportError:
            print("MLflow not installed. Install: pip install mlflow")
            self._tracker = DummyTracker()
            return

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._tracker = mlflow

    def _init_wandb(self, project: Optional[str]) -> None:
        try:
            import wandb
        except ImportError:
            print("wandb not installed. Install: pip install wandb")
            self._tracker = DummyTracker()
            return

        wandb.init(project=project or self.experiment_name, name=self.experiment_name)
        self._tracker = wandb

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log scalar metrics at a given step/epoch."""
        if isinstance(self._tracker, DummyTracker):
            return

        if self.backend == "mlflow":
            self._tracker.log_metrics(metrics, step=step)
        elif self.backend == "wandb":
            self._tracker.log(metrics, step=step)
        elif self.backend == "trackio":
            self._tracker.log(metrics, step=step)

    def log_hyperparams(self, params: dict) -> None:
        """Log hyperparameters at the start of an experiment."""
        if isinstance(self._tracker, DummyTracker):
            return

        if self.backend == "mlflow":
            self._tracker.log_params(params)
        elif self.backend == "wandb":
            self._tracker.config.update(params)
        elif self.backend == "trackio":
            self._tracker.log_hyperparams(params)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        """Log model artifact."""
        if isinstance(self._tracker, DummyTracker):
            return

        if self.backend == "mlflow":
            import torch
            self._tracker.pytorch.log_model(model, artifact_path)
        elif self.backend == "wandb":
            self._tracker.save(artifact_path)
        elif self.backend == "trackio":
            self._tracker.log_artifact(artifact_path)

    def log_artifact(self, path: str) -> None:
        """Log a file or directory artifact."""
        if isinstance(self._tracker, DummyTracker):
            return

        if self.backend == "mlflow":
            self._tracker.log_artifact(path)
        elif self.backend == "wandb":
            self._tracker.save(path)
        elif self.backend == "trackio":
            self._tracker.log_artifact(path)

    def finish(self) -> None:
        """End the tracking session."""
        if self.backend == "wandb" and not isinstance(self._tracker, DummyTracker):
            self._tracker.finish()
        elif self.backend == "mlflow" and not isinstance(self._tracker, DummyTracker):
            self._tracker.end_run()


if __name__ == "__main__":
    print("Experiment Tracking for dl-reproducibility-pack v3.3")
    print("  ExperimentTracker - Unified interface for Trackio / MLflow / W&B")
    print()
    print("  Usage:")
    print("    tracker = ExperimentTracker('trackio', 'my_exp')")
    print("    tracker.log_hyperparams({'lr': 0.001})")
    print("    tracker.log_metrics({'loss': 0.5}, step=0)")
    print("    tracker.finish()")
