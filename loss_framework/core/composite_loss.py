"""
Composite Loss Module
Implements Composite pattern for combining multiple loss functions
Enables weighted combination of multiple loss terms
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from .base_loss import BaseLoss


class CompositeLoss(BaseLoss):
    """
    Composite pattern implementation for combining multiple loss functions.

    Allows weighted combination of multiple loss terms:
        L_total = w1*L1 + w2*L2 + ... + wn*Ln

    Supports:
    - Weighted combination of losses
    - Dynamic weight adjustment
    - Individual loss tracking
    - Gradient balancing

    Example:
        composite = CompositeLoss({
            'mse': MSELoss(),
            'l1': L1Loss()
        }, weights={'mse': 0.7, 'l1': 0.3})

        loss = composite(predictions, targets)
    """

    def __init__(
        self,
        losses: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None,
        normalize_weights: bool = False,
        reduction: str = "mean",
        device: str = "auto",
    ):
        """
        Initialize composite loss.

        Args:
            losses: Dictionary mapping loss names to loss instances
            weights: Optional dictionary mapping loss names to weights
            normalize_weights: Whether to normalize weights to sum to 1
            reduction: Reduction method (applied to final combined loss)
            device: Device for computation
        """
        super().__init__(reduction=reduction, device=device)

        self.losses = nn.ModuleDict(losses)
        self.loss_names = list(losses.keys())

        # Set default weights if not provided
        if weights is None:
            weights = {name: 1.0 for name in self.loss_names}

        self.weights = weights
        self.normalize_weights = normalize_weights

        # Validate weights
        self._validate_weights()

        # Individual loss tracking
        self._loss_history: Dict[str, List[float]] = {
            name: [] for name in self.loss_names
        }
        self._current_losses: Dict[str, float] = {}

        # Register weights as buffers for device management
        self._register_weights()

    def _validate_weights(self) -> None:
        """Validate that all losses have corresponding weights."""
        for name in self.loss_names:
            if name not in self.weights:
                raise ValueError(f"No weight provided for loss: {name}")

        # Check for extra weights
        for name in self.weights:
            if name not in self.loss_names:
                raise ValueError(f"Weight provided for unknown loss: {name}")

    def _register_weights(self) -> None:
        """Register weights as buffers for automatic device management."""
        weight_tensor = torch.tensor(
            [self.weights[name] for name in self.loss_names], dtype=torch.float32
        )

        if self.normalize_weights:
            weight_tensor = weight_tensor / weight_tensor.sum()

        self.register_buffer("_weight_tensor", weight_tensor)

    def _get_normalized_weights(self) -> torch.Tensor:
        """Get normalized weights."""
        if self.normalize_weights:
            return self._weight_tensor / self._weight_tensor.sum()
        return self._weight_tensor

    def _compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Compute composite loss as weighted sum of individual losses.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments passed to individual losses

        Returns:
            Combined loss tensor
        """
        total_loss = torch.tensor(0.0, device=predictions.device)
        weights = self._get_normalized_weights()

        self._current_losses = {}

        for idx, (name, loss_fn) in enumerate(self.losses.items()):
            # Compute individual loss
            individual_loss = loss_fn(predictions, targets, **kwargs)

            # Handle different return types
            if isinstance(individual_loss, dict):
                # Some losses return dict with multiple terms
                loss_value = individual_loss.get(
                    "loss", individual_loss.get("total", 0)
                )
            else:
                loss_value = individual_loss

            # Weight the loss
            weighted_loss = weights[idx] * loss_value

            # Accumulate
            total_loss = total_loss + weighted_loss

            # Track individual loss
            with torch.no_grad():
                self._current_losses[name] = (
                    loss_value.item() if torch.is_tensor(loss_value) else loss_value
                )
                self._loss_history[name].append(self._current_losses[name])

        return total_loss

    def get_individual_losses(self) -> Dict[str, float]:
        """Get individual loss values from last forward pass."""
        return self._current_losses.copy()

    def get_loss_history(self, name: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Get loss history.

        Args:
            name: Optional loss name to get history for specific loss

        Returns:
            Dictionary mapping loss names to their history
        """
        if name is not None:
            return {name: self._loss_history[name]}
        return {k: v.copy() for k, v in self._loss_history.items()}

    def set_weight(self, name: str, weight: float) -> None:
        """
        Update weight for a specific loss.

        Args:
            name: Name of the loss
            weight: New weight value
        """
        if name not in self.loss_names:
            raise ValueError(f"Unknown loss: {name}")

        self.weights[name] = weight
        self._register_weights()  # Re-register with new weights

    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        weights_tensor = self._get_normalized_weights()
        return {
            name: weights_tensor[idx].item() for idx, name in enumerate(self.loss_names)
        }

    def reset_history(self) -> None:
        """Reset loss history."""
        self._loss_history = {name: [] for name in self.loss_names}
        self._current_losses = {}

    def add_loss(self, name: str, loss: nn.Module, weight: float = 1.0) -> None:
        """
        Add a new loss to the composite.

        Args:
            name: Name for the new loss
            loss: Loss function instance
            weight: Weight for the new loss
        """
        if name in self.loss_names:
            raise ValueError(f"Loss with name '{name}' already exists")

        self.losses[name] = loss
        self.loss_names.append(name)
        self.weights[name] = weight
        self._loss_history[name] = []

        self._register_weights()

    def remove_loss(self, name: str) -> None:
        """
        Remove a loss from the composite.

        Args:
            name: Name of loss to remove
        """
        if name not in self.loss_names:
            raise ValueError(f"Unknown loss: {name}")

        del self.losses[name]
        del self.weights[name]
        del self._loss_history[name]
        self.loss_names.remove(name)

        if name in self._current_losses:
            del self._current_losses[name]

        self._register_weights()

    def extra_repr(self) -> str:
        """String representation."""
        weights_str = ", ".join([f"{k}={v:.3f}" for k, v in self.get_weights().items()])
        return f"losses={list(self.losses.keys())}, weights={{{weights_str}}}"

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "losses": list(self.loss_names),
            "weights": self.get_weights(),
            "normalize_weights": self.normalize_weights,
            "reduction": self.reduction,
        }
