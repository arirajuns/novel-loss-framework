"""
Base Loss Module
Implements Template Method pattern for loss functions
Provides common structure and hooks for customization
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np


class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for all loss functions using Template Method pattern.

    This class defines the skeleton of a loss function algorithm,
    deferring some steps to subclasses.

    Template Method Pattern Structure:
    1. forward() - Template method defining the algorithm structure
    2. _preprocess_inputs() - Hook for input validation/preprocessing
    3. _compute_loss() - Abstract method for actual loss computation (subclass responsibility)
    4. _postprocess_loss() - Hook for loss post-processing
    5. _compute_gradients() - Hook for custom gradient computation
    """

    def __init__(
        self,
        reduction: str = "mean",
        weight: Optional[torch.Tensor] = None,
        device: str = "auto",
    ):
        """
        Initialize base loss.

        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            weight: Optional weight tensor for weighted losses
            device: Device to use ('auto', 'cpu', 'cuda', etc.)
        """
        super().__init__()

        self.reduction = reduction
        self.weight = weight
        self.device = self._get_device(device)

        # Statistics tracking
        self._statistics = {
            "call_count": 0,
            "total_loss": 0.0,
            "min_loss": float("inf"),
            "max_loss": float("-inf"),
        }

        # Gradient tracking
        self._track_gradients = False
        self._gradient_stats = {}

    def _get_device(self, device: str) -> torch.device:
        """Determine device to use."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Template method defining the loss computation algorithm.

        This method defines the skeleton and calls various hook methods
        that can be overridden by subclasses.

        Algorithm:
        1. Validate inputs
        2. Preprocess inputs (hook)
        3. Compute loss (abstract - subclass implements)
        4. Apply reduction
        5. Postprocess loss (hook)
        6. Update statistics
        7. Return loss
        """
        # Step 1: Validate inputs
        self._validate_inputs(predictions, targets)

        # Move to device FIRST (before preprocessing)
        predictions = predictions.to(self.device)
        targets = targets.to(self.device)

        # Step 2: Preprocess inputs (hook)
        predictions, targets = self._preprocess_inputs(predictions, targets)

        # Step 3: Compute loss (abstract method)
        loss = self._compute_loss(predictions, targets, **kwargs)

        # Step 4: Apply reduction
        loss = self._apply_reduction(loss)

        # Step 5: Postprocess loss (hook)
        loss = self._postprocess_loss(loss)

        # Step 6: Update statistics
        self._update_statistics(loss)

        return loss

    def _validate_inputs(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Validate input tensors."""
        if not isinstance(predictions, torch.Tensor):
            raise TypeError(
                f"predictions must be a torch.Tensor, got {type(predictions)}"
            )

        if not isinstance(targets, torch.Tensor):
            raise TypeError(f"targets must be a torch.Tensor, got {type(targets)}")

        if predictions.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Batch size mismatch: predictions {predictions.shape[0]} "
                f"vs targets {targets.shape[0]}"
            )

    def _preprocess_inputs(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hook method for input preprocessing.
        Subclasses can override to add custom preprocessing.

        Default: No preprocessing (identity operation)
        """
        return predictions, targets

    @abstractmethod
    def _compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Abstract method for computing the loss.
        MUST be implemented by all subclasses.

        This is the core algorithm step that varies between loss functions.
        """
        pass

    def _apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to loss tensor."""
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

    def _postprocess_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Hook method for loss postprocessing.
        Subclasses can override to add custom postprocessing.

        Default: Apply weight if provided
        """
        if self.weight is not None:
            loss = loss * self.weight.to(loss.device)
        return loss

    def _update_statistics(self, loss: torch.Tensor) -> None:
        """Update loss statistics."""
        with torch.no_grad():
            loss_val = loss.item() if loss.numel() == 1 else loss.mean().item()

            self._statistics["call_count"] += 1
            self._statistics["total_loss"] += loss_val
            self._statistics["min_loss"] = min(self._statistics["min_loss"], loss_val)
            self._statistics["max_loss"] = max(self._statistics["max_loss"], loss_val)

    def get_statistics(self) -> Dict[str, Any]:
        """Get loss statistics."""
        stats = self._statistics.copy()
        if stats["call_count"] > 0:
            stats["avg_loss"] = stats["total_loss"] / stats["call_count"]
        return stats

    def reset_statistics(self) -> None:
        """Reset loss statistics."""
        self._statistics = {
            "call_count": 0,
            "total_loss": 0.0,
            "min_loss": float("inf"),
            "max_loss": float("-inf"),
        }

    def enable_gradient_tracking(self) -> None:
        """Enable gradient statistics tracking."""
        self._track_gradients = True

    def disable_gradient_tracking(self) -> None:
        """Disable gradient statistics tracking."""
        self._track_gradients = False

    @property
    def name(self) -> str:
        """Get loss function name."""
        return self.__class__.__name__

    def extra_repr(self) -> str:
        """String representation for printing."""
        return f"reduction={self.reduction}, device={self.device}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert loss configuration to dictionary."""
        return {
            "name": self.name,
            "reduction": self.reduction,
            "device": str(self.device),
            "has_weight": self.weight is not None,
        }
