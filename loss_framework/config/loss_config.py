"""
Loss Configuration Module
Implements Builder pattern for constructing loss function configurations
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from .base_config import BaseConfig


@dataclass
class LossConfig(BaseConfig):
    """
    Configuration for loss functions using Builder pattern.
    Provides flexible configuration with sensible defaults.
    """

    # Required fields
    loss_type: str = "cross_entropy"

    # Loss-specific hyperparameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Reduction method
    reduction: str = "mean"  # Options: 'mean', 'sum', 'none'

    # Device configuration
    device: str = "auto"  # Options: 'auto', 'cpu', 'cuda', 'cuda:0', etc.

    # Weight configuration for weighted losses
    class_weights: Optional[List[float]] = None

    # Loss scaling factor
    loss_scale: float = 1.0

    # Gradient clipping
    gradient_clip_val: Optional[float] = None

    # Label smoothing (for classification losses)
    label_smoothing: float = 0.0

    # Loss-specific flags
    ignore_index: int = -100

    def validate(self) -> None:
        """Validate loss configuration parameters."""
        valid_reductions = ["mean", "sum", "none"]
        if self.reduction not in valid_reductions:
            raise ValueError(
                f"Invalid reduction: {self.reduction}. "
                f"Must be one of {valid_reductions}"
            )

        if self.loss_scale <= 0:
            raise ValueError(f"loss_scale must be positive, got {self.loss_scale}")

        if self.label_smoothing < 0 or self.label_smoothing > 1:
            raise ValueError(
                f"label_smoothing must be in [0, 1], got {self.label_smoothing}"
            )

        if self.gradient_clip_val is not None and self.gradient_clip_val <= 0:
            raise ValueError(
                f"gradient_clip_val must be positive, got {self.gradient_clip_val}"
            )


@dataclass
class AdaptiveLossConfig(LossConfig):
    """Configuration for Adaptive Weighted Loss."""

    loss_type: str = "adaptive_weighted"

    # Initial weight
    initial_weight: float = 1.0

    # Weight schedule type
    schedule_type: str = "exponential"  # 'linear', 'exponential', 'cosine'

    # Schedule parameters
    warmup_epochs: int = 0
    decay_epochs: int = 100
    min_weight: float = 0.1
    max_weight: float = 10.0

    # Curriculum learning
    use_curriculum: bool = False
    difficulty_threshold: float = 0.5

    def validate(self) -> None:
        """Validate adaptive loss configuration."""
        super().validate()

        valid_schedules = ["linear", "exponential", "cosine"]
        if self.schedule_type not in valid_schedules:
            raise ValueError(f"Invalid schedule_type: {self.schedule_type}")

        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be non-negative")

        if self.decay_epochs <= 0:
            raise ValueError(f"decay_epochs must be positive")


@dataclass
class GeometricLossConfig(LossConfig):
    """Configuration for Geometric Distance Loss."""

    loss_type: str = "geometric"

    # Manifold parameters
    manifold_type: str = "euclidean"  # 'euclidean', 'spherical', 'hyperbolic'

    # Distance metric
    distance_metric: str = "geodesic"  # 'geodesic', 'chordal', 'embedding'

    # Embedding dimension
    embedding_dim: int = 128

    # Curvature parameter (for non-Euclidean manifolds)
    curvature: float = 1.0

    # Projection parameters
    project_to_manifold: bool = True

    def validate(self) -> None:
        """Validate geometric loss configuration."""
        super().validate()

        valid_manifolds = ["euclidean", "spherical", "hyperbolic"]
        if self.manifold_type not in valid_manifolds:
            raise ValueError(f"Invalid manifold_type: {self.manifold_type}")

        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive")

        if self.curvature <= 0 and self.manifold_type != "euclidean":
            raise ValueError(f"curvature must be positive for non-Euclidean manifolds")


@dataclass
class InformationTheoreticLossConfig(LossConfig):
    """Configuration for Information-Theoretic Loss."""

    loss_type: str = "information_theoretic"

    # Information-theoretic components
    use_entropy_regularization: bool = True
    entropy_weight: float = 0.1

    use_mutual_information: bool = False
    mi_weight: float = 0.1

    # KL divergence parameters
    use_kl_divergence: bool = False
    kl_weight: float = 0.1

    # Temperature parameter for soft distributions
    temperature: float = 1.0

    def validate(self) -> None:
        """Validate information-theoretic loss configuration."""
        super().validate()

        if self.entropy_weight < 0:
            raise ValueError(f"entropy_weight must be non-negative")

        if self.mi_weight < 0:
            raise ValueError(f"mi_weight must be non-negative")

        if self.kl_weight < 0:
            raise ValueError(f"kl_weight must be non-negative")

        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive")


@dataclass
class PhysicsInspiredLossConfig(LossConfig):
    """Configuration for Physics-Inspired Loss."""

    loss_type: str = "physics_inspired"

    # Hamiltonian parameters
    use_hamiltonian: bool = True
    hamiltonian_weight: float = 0.1

    # Conservation law parameters
    use_conservation: bool = False
    conservation_weight: float = 0.1
    conserved_quantities: int = 1

    # Lagrangian parameters
    use_lagrangian: bool = False
    lagrangian_weight: float = 0.1

    # Symplectic integration
    use_symplectic: bool = False
    symplectic_order: int = 2

    def validate(self) -> None:
        """Validate physics-inspired loss configuration."""
        super().validate()

        if self.hamiltonian_weight < 0:
            raise ValueError(f"hamiltonian_weight must be non-negative")

        if self.conservation_weight < 0:
            raise ValueError(f"conservation_weight must be non-negative")

        if self.lagrangian_weight < 0:
            raise ValueError(f"lagrangian_weight must be non-negative")

        if self.conserved_quantities <= 0:
            raise ValueError(f"conserved_quantities must be positive")


@dataclass
class RobustStatisticalLossConfig(LossConfig):
    """Configuration for Robust Statistical Loss."""

    loss_type: str = "robust_statistical"

    # Robust loss type
    robust_type: str = "huber"  # 'huber', 'tukey', 'cauchy', 'geman_mcclure'

    # Scale parameter
    scale: float = 1.0

    # Adaptive scale
    adaptive_scale: bool = False
    scale_update_rate: float = 0.1

    # Outlier handling
    outlier_threshold: Optional[float] = None

    def validate(self) -> None:
        """Validate robust statistical loss configuration."""
        super().validate()

        valid_types = ["huber", "tukey", "cauchy", "geman_mcclure"]
        if self.robust_type not in valid_types:
            raise ValueError(f"Invalid robust_type: {self.robust_type}")

        if self.scale <= 0:
            raise ValueError(f"scale must be positive")

        if self.scale_update_rate <= 0 or self.scale_update_rate > 1:
            raise ValueError(f"scale_update_rate must be in (0, 1]")
