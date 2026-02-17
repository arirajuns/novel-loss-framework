"""
Experiment Configuration Module
Comprehensive experiment settings using Builder pattern
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from .base_config import BaseConfig
from .loss_config import LossConfig


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for model architecture."""

    model_name: str = "simple_nn"
    input_dim: int = 784
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    output_dim: int = 10
    dropout_rate: float = 0.2
    activation: str = "relu"
    use_batch_norm: bool = True

    def validate(self) -> None:
        """Validate model configuration."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive")
        if any(h <= 0 for h in self.hidden_dims):
            raise ValueError(f"All hidden_dims must be positive")
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError(f"dropout_rate must be in [0, 1]")


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training process."""

    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "adam"  # 'adam', 'sgd', 'rmsprop', 'adamw'
    weight_decay: float = 0.0001
    momentum: float = 0.9  # For SGD
    scheduler: Optional[str] = "cosine"  # 'step', 'cosine', 'plateau', None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    def validate(self) -> None:
        """Validate training configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative")


@dataclass
class DataConfig(BaseConfig):
    """Configuration for data loading and preprocessing."""

    dataset_name: str = "mnist"
    data_dir: str = "./data"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    normalize: bool = True
    augmentation: bool = False
    random_seed: int = 42

    def validate(self) -> None:
        """Validate data configuration."""
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative")


@dataclass
class LoggingConfig(BaseConfig):
    """Configuration for experiment logging."""

    log_dir: str = "./experiments/logs"
    log_level: str = "INFO"
    use_tensorboard: bool = True
    tensorboard_dir: str = "./experiments/tensorboard"
    use_wandb: bool = False
    wandb_project: str = "loss_framework"
    wandb_entity: Optional[str] = None
    log_frequency: int = 10  # Log every N batches
    save_frequency: int = 1  # Save checkpoint every N epochs
    keep_last_n: int = 3  # Keep only last N checkpoints

    def validate(self) -> None:
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ValueError(f"Invalid log_level: {self.log_level}")
        if self.log_frequency <= 0:
            raise ValueError(f"log_frequency must be positive")


@dataclass
class EvaluationConfig(BaseConfig):
    """Configuration for model evaluation."""

    metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss"])
    eval_frequency: int = 1  # Evaluate every N epochs
    visualize_gradients: bool = True
    visualize_loss_landscape: bool = False
    save_predictions: bool = False
    confusion_matrix: bool = True

    def validate(self) -> None:
        """Validate evaluation configuration."""
        if self.eval_frequency <= 0:
            raise ValueError(f"eval_frequency must be positive")


@dataclass
class ExperimentConfig(BaseConfig):
    """
    Master experiment configuration aggregating all sub-configurations.
    Uses composition pattern to combine different configuration types.
    """

    # Experiment identification
    experiment_name: str = "experiment_001"
    experiment_id: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Sub-configurations
    loss_config: LossConfig = field(default_factory=LossConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Experiment metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    git_commit: Optional[str] = None

    def validate(self) -> None:
        """Validate all sub-configurations."""
        # Validate all sub-configurations
        self.loss_config.validate()
        self.model_config.validate()
        self.training_config.validate()
        self.data_config.validate()
        self.logging_config.validate()
        self.evaluation_config.validate()

    def get_experiment_dir(self) -> Path:
        """Get experiment directory path."""
        exp_dir = (
            Path(self.logging_config.log_dir)
            / f"{self.experiment_name}_{self.experiment_id}"
        )
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def save(self, filepath: Optional[str] = None) -> str:
        """Save full experiment configuration."""
        if filepath is None:
            filepath = self.get_experiment_dir() / "config.yaml"

        return self.to_yaml(filepath)

    @classmethod
    def load(cls, filepath: str) -> "ExperimentConfig":
        """Load experiment configuration."""
        return cls.from_yaml(filepath)

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ExperimentConfig(\n"
            f"  name={self.experiment_name},\n"
            f"  id={self.experiment_id},\n"
            f"  loss={self.loss_config.loss_type},\n"
            f"  model={self.model_config.model_name},\n"
            f"  epochs={self.training_config.num_epochs}\n"
            f")"
        )
