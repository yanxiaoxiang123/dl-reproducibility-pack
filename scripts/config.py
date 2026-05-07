"""
config.py - Configuration Management for Deep Learning Experiments

Provides a simple yet powerful configuration system for managing hyperparameters,
experiment settings, and reproducibility parameters in YAML files.

Usage:
    from src.config import Config, load_config

    config = load_config("configs/default.yaml")
    print(config.training.batch_size)

Author: dl-reproducibility-pack
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Union, Any
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for experiment metadata and reproducibility."""
    name: str = "default_experiment"
    seed: int = 42
    deterministic: bool = True
    output_dir: str = "outputs/default"
    save_checkpoints: bool = True
    resume_from: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    name: str = "ResNet50"
    pretrained: bool = False
    num_classes: int = 1000
    dropout: float = 0.0
    checkpoint: Optional[str] = None


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    dataset: str = "CIFAR100"
    data_dir: str = "./data"
    image_size: int = 224
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    augmentation: bool = True
    num_classes: int = 100
    val_split: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 0.001
    min_lr: float = 0.0
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"
    warmup_epochs: int = 5
    warmup_steps: int = 0
    weight_decay: float = 0.0001
    momentum: float = 0.9
    betas: tuple = (0.9, 0.999)
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = False
    accumulation_steps: int = 1
    label_smoothing: float = 0.0
    ema_decay: float = 0.0


@dataclass
class HardwareConfig:
    """Configuration for hardware and compute resources."""
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging and experiment tracking."""
    log_interval: int = 10
    eval_interval: int = 1
    save_checkpoint_interval: int = 10
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "my-research"
    wandb_entity: Optional[str] = None


@dataclass
class Config:
    """Main configuration container for deep learning experiments."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            Config object populated from the file

        Example:
            >>> config = Config.from_yaml("configs/default.yaml")
            >>> print(config.training.batch_size)
            128
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        """Create Config from a dictionary."""
        return cls(
            experiment=ExperimentConfig(**data.get("experiment", {})),
            model=ModelConfig(**data.get("model", {})),
            data=DataConfig(**data.get("data", {})),
            training=TrainingConfig(**data.get("training", {})),
            hardware=HardwareConfig(**data.get("hardware", {})),
            logging=LoggingConfig(**data.get("logging", {})),
        )

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.

        Args:
            path: Path where to save the YAML file

        Example:
            >>> config.to_yaml("configs/my_experiment.yaml")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "experiment": self.experiment.__dict__,
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "hardware": self.hardware.__dict__,
            "logging": self.logging.__dict__,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def update(self, **kwargs) -> "Config":
        """
        Create a new Config with updated values.

        Args:
            **kwargs: Values to update, prefixed with section name

        Example:
            >>> new_config = config.update(**{
            ...     "training.batch_size": 256,
            ...     "experiment.seed": 12345
            ... })
        """
        data = {
            "experiment": self.experiment.__dict__,
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "hardware": self.hardware.__dict__,
            "logging": self.logging.__dict__,
        }

        for key, value in kwargs.items():
            if "." in key:
                section, field = key.split(".", 1)
                if section in data and field in data[section]:
                    data[section][field] = value

        return self._from_dict(data)

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-separated path.

        Args:
            path: Dot-separated path like "training.batch_size"
            default: Default value if path not found

        Example:
            >>> config.get("training.batch_size")
            128
            >>> config.get("training.nonexistent", "default")
            'default'
        """
        if "." in path:
            section, field = path.split(".", 1)
            data = {
                "experiment": self.experiment.__dict__,
                "model": self.model.__dict__,
                "data": self.data.__dict__,
                "training": self.training.__dict__,
                "hardware": self.hardware.__dict__,
                "logging": self.logging.__dict__,
            }
            if section in data and field in data[section]:
                return data[section][field]
        return default


def load_config(path: Union[str, Path]) -> Config:
    """
    Convenience function to load a configuration from YAML.

    Args:
        path: Path to the YAML configuration file

    Returns:
        Config object

    Example:
        >>> config = load_config("configs/default.yaml")
    """
    return Config.from_yaml(path)


def create_default_config(output_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Create a Config with default values, optionally saving to file.

    Args:
        output_path: Optional path to save the default config

    Returns:
        Config object with default values

    Example:
        >>> config = create_default_config()
        >>> config = create_default_config("configs/default.yaml")
    """
    config = Config()
    if output_path:
        config.to_yaml(output_path)
    return config


if __name__ == "__main__":
    # Example usage when run directly
    print("Creating default config...")
    config = create_default_config()

    print("\nDefault config values:")
    print(f"  Experiment: {config.experiment.name}")
    print(f"  Seed: {config.experiment.seed}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  GPU IDs: {config.hardware.gpu_ids}")

    print("\nSaving to configs/default.yaml...")
    config.to_yaml("configs/default.yaml")
    print("Done!")
