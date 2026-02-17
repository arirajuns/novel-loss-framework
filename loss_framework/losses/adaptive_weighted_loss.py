"""
Adaptive Weighted Loss
Implements curriculum learning with dynamic weight adjustment
Uses Strategy pattern for different scheduling algorithms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math

from ..core.base_loss import BaseLoss
from ..core.loss_registry import register_loss
from ..config.loss_config import AdaptiveLossConfig


class WeightScheduleStrategy:
    """
    Strategy pattern for weight scheduling algorithms.
    Defines interface for different scheduling strategies.
    """

    @staticmethod
    def linear_schedule(
        epoch: int,
        warmup_epochs: int,
        decay_epochs: int,
        min_weight: float,
        max_weight: float,
        initial_weight: float,
    ) -> float:
        """Linear weight schedule."""
        if epoch < warmup_epochs:
            return initial_weight + (max_weight - initial_weight) * (
                epoch / warmup_epochs
            )
        elif epoch < warmup_epochs + decay_epochs:
            progress = (epoch - warmup_epochs) / decay_epochs
            return max_weight - (max_weight - min_weight) * progress
        else:
            return min_weight

    @staticmethod
    def exponential_schedule(
        epoch: int,
        warmup_epochs: int,
        decay_epochs: int,
        min_weight: float,
        max_weight: float,
        initial_weight: float,
    ) -> float:
        """Exponential weight schedule."""
        if epoch < warmup_epochs:
            # At epoch 0: multiplier should be 0, so result = initial_weight
            # At epoch = warmup_epochs: multiplier should be 1, so result = max_weight
            if warmup_epochs > 0:
                multiplier = 1 - math.exp(-5 * epoch / warmup_epochs)
            else:
                multiplier = 1.0
            return initial_weight + (max_weight - initial_weight) * multiplier
        elif epoch < warmup_epochs + decay_epochs:
            progress = (epoch - warmup_epochs) / decay_epochs
            return min_weight + (max_weight - min_weight) * math.exp(-5 * progress)
        else:
            return min_weight

    @staticmethod
    def cosine_schedule(
        epoch: int,
        warmup_epochs: int,
        decay_epochs: int,
        min_weight: float,
        max_weight: float,
        initial_weight: float,
    ) -> float:
        """Cosine annealing weight schedule."""
        if epoch < warmup_epochs:
            return initial_weight + (max_weight - initial_weight) * (
                0.5 * (1 - math.cos(math.pi * epoch / warmup_epochs))
            )
        elif epoch < warmup_epochs + decay_epochs:
            progress = (epoch - warmup_epochs) / decay_epochs
            return min_weight + (max_weight - min_weight) * (
                0.5 * (1 + math.cos(math.pi * progress))
            )
        else:
            return min_weight


@register_loss(name="adaptive_weighted", category="adaptive")
class AdaptiveWeightedLoss(BaseLoss):
    """
    Adaptive Weighted Loss with curriculum learning.

    Dynamically adjusts loss weight based on training progress.
    Implements curriculum learning by focusing on difficult examples.

    Mathematical Formulation:
        L_total = w(t) * L_base + (1 - w(t)) * L_curriculum

    Where:
        - w(t) is the time-dependent weight
        - L_base is the base loss (e.g., cross-entropy)
        - L_curriculum is the curriculum loss for hard examples

    Features:
    - Multiple scheduling strategies (linear, exponential, cosine)
    - Difficulty-based example weighting
    - Automatic curriculum adjustment
    - Statistics tracking

    Example:
        config = AdaptiveLossConfig(
            loss_type="adaptive_weighted",
            initial_weight=1.0,
            schedule_type="cosine",
            warmup_epochs=10,
            decay_epochs=90
        )
        loss = AdaptiveWeightedLoss.from_config(config)
    """

    def __init__(
        self,
        base_loss: str = "cross_entropy",
        initial_weight: float = 1.0,
        schedule_type: str = "exponential",
        warmup_epochs: int = 0,
        decay_epochs: int = 100,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
        use_curriculum: bool = False,
        difficulty_threshold: float = 0.5,
        reduction: str = "mean",
        device: str = "auto",
    ):
        """
        Initialize adaptive weighted loss.

        Args:
            base_loss: Base loss function type
            initial_weight: Initial loss weight
            schedule_type: Weight schedule type ('linear', 'exponential', 'cosine')
            warmup_epochs: Number of warmup epochs
            decay_epochs: Number of decay epochs
            min_weight: Minimum weight value
            max_weight: Maximum weight value
            use_curriculum: Whether to use curriculum learning
            difficulty_threshold: Threshold for difficulty-based weighting
            reduction: Loss reduction method
            device: Device for computation
        """
        super().__init__(reduction=reduction, device=device)

        self.base_loss_type = base_loss
        self.initial_weight = initial_weight
        self.schedule_type = schedule_type
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.use_curriculum = use_curriculum
        self.difficulty_threshold = difficulty_threshold

        # Current state
        self.current_epoch = 0
        self.current_weight = initial_weight

        # Select scheduling strategy
        self.schedule_fn = self._get_schedule_function()

        # Create base loss
        self.base_loss_fn = self._create_base_loss()

        # Curriculum tracking
        self._difficulty_history = []

    def _get_schedule_function(self):
        """Get the scheduling function based on type."""
        schedules = {
            "linear": WeightScheduleStrategy.linear_schedule,
            "exponential": WeightScheduleStrategy.exponential_schedule,
            "cosine": WeightScheduleStrategy.cosine_schedule,
        }

        if self.schedule_type not in schedules:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        return schedules[self.schedule_type]

    def _create_base_loss(self):
        """Create base loss function."""
        loss_map = {
            "cross_entropy": nn.CrossEntropyLoss(reduction="none"),
            "mse": nn.MSELoss(reduction="none"),
            "l1": nn.L1Loss(reduction="none"),
            "bce": nn.BCELoss(reduction="none"),
        }

        if self.base_loss_type not in loss_map:
            raise ValueError(f"Unknown base loss: {self.base_loss_type}")

        return loss_map[self.base_loss_type].to(self.device)

    def update_epoch(self, epoch: int) -> None:
        """
        Update current epoch and recalculate weight.

        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch
        self.current_weight = self.schedule_fn(
            epoch,
            self.warmup_epochs,
            self.decay_epochs,
            self.min_weight,
            self.max_weight,
            self.initial_weight,
        )

    def _preprocess_inputs(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Preprocess inputs with curriculum learning."""
        if self.use_curriculum:
            # Calculate per-example difficulty
            with torch.no_grad():
                base_losses = self.base_loss_fn(predictions, targets)
                difficulties = (base_losses > self.difficulty_threshold).float()
                self._difficulty_history.append(difficulties.mean().item())

            # Store curriculum weights for later use
            self._curriculum_weights = 1.0 + difficulties
        else:
            self._curriculum_weights = None

        return predictions, targets

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute adaptive weighted loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Weighted loss
        """
        # Compute base loss per element
        base_loss = self.base_loss_fn(predictions, targets)

        # Apply curriculum weighting if enabled
        if self._curriculum_weights is not None:
            base_loss = base_loss * self._curriculum_weights

        # Apply adaptive weight
        weighted_loss = self.current_weight * base_loss

        return weighted_loss

    def get_current_weight(self) -> float:
        """Get current adaptive weight."""
        return self.current_weight

    def get_difficulty_stats(self) -> Dict[str, float]:
        """Get curriculum difficulty statistics."""
        if not self._difficulty_history:
            return {}

        import numpy as np

        diffs = self._difficulty_history
        return {
            "mean_difficulty": np.mean(diffs),
            "std_difficulty": np.std(diffs),
            "min_difficulty": np.min(diffs),
            "max_difficulty": np.max(diffs),
        }

    @classmethod
    def from_config(cls, config: AdaptiveLossConfig) -> "AdaptiveWeightedLoss":
        """Create loss from configuration."""
        return cls(
            base_loss=config.hyperparameters.get("base_loss", "cross_entropy"),
            initial_weight=config.initial_weight,
            schedule_type=config.schedule_type,
            warmup_epochs=config.warmup_epochs,
            decay_epochs=config.decay_epochs,
            min_weight=config.min_weight,
            max_weight=config.max_weight,
            use_curriculum=config.use_curriculum,
            difficulty_threshold=config.difficulty_threshold,
            reduction=config.reduction,
            device=config.device,
        )

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"base_loss={self.base_loss_type}, "
            f"schedule={self.schedule_type}, "
            f"current_weight={self.current_weight:.4f}, "
            f"curriculum={self.use_curriculum}"
        )


class DynamicFocalLoss(BaseLoss):
    """
    Dynamic Focal Loss with adaptive focusing parameter.

    Extends standard focal loss with dynamic gamma adjustment.
    Automatically adjusts focusing based on training progress.

    Mathematical Formulation:
        FL(pt) = -αt * (1 - pt)^γ * log(pt)

    Where:
        - pt is the predicted probability for the true class
        - αt is the class weighting factor
        - γ is the focusing parameter (adaptive)

    Features:
    - Dynamic gamma adjustment
    - Class-balanced weighting
    - Asymptotic focusing
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        initial_gamma: float = 2.0,
        final_gamma: float = 5.0,
        warmup_epochs: int = 10,
        reduction: str = "mean",
        device: str = "auto",
    ):
        """
        Initialize dynamic focal loss.

        Args:
            alpha: Class weights
            initial_gamma: Initial focusing parameter
            final_gamma: Final focusing parameter
            warmup_epochs: Epochs to reach final gamma
            reduction: Loss reduction method
            device: Device for computation
        """
        super().__init__(reduction=reduction, device=device)

        self.alpha = alpha.to(device) if alpha is not None else None
        self.initial_gamma = initial_gamma
        self.final_gamma = final_gamma
        self.warmup_epochs = warmup_epochs
        self.current_gamma = initial_gamma
        self.current_epoch = 0

    def update_epoch(self, epoch: int) -> None:
        """Update gamma based on training progress."""
        self.current_epoch = epoch

        if self.warmup_epochs > 0:
            progress = min(epoch / self.warmup_epochs, 1.0)
            self.current_gamma = (
                self.initial_gamma + (self.final_gamma - self.initial_gamma) * progress
            )

    def _compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute dynamic focal loss."""
        # Apply softmax to get probabilities
        probs = F.softmax(predictions, dim=1)

        # Get probability of true class
        batch_size = predictions.size(0)
        pt = probs[torch.arange(batch_size), targets]

        # Compute focal weight
        focal_weight = (1 - pt) ** self.current_gamma

        # Compute cross-entropy
        ce_loss = F.cross_entropy(
            predictions, targets, reduction="none", weight=self.alpha
        )

        # Apply focal weighting
        focal_loss = focal_weight * ce_loss

        return focal_loss

    def get_current_gamma(self) -> float:
        """Get current gamma value."""
        return self.current_gamma

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"gamma={self.current_gamma:.2f}, "
            f"initial_gamma={self.initial_gamma}, "
            f"final_gamma={self.final_gamma}"
        )
