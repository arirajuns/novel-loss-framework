"""
Robust Statistical Loss
Implements robust loss functions less sensitive to outliers
Uses M-estimators and adaptive weighting from robust statistics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from ..core.base_loss import BaseLoss
from ..core.loss_registry import register_loss
from ..config.loss_config import RobustStatisticalLossConfig


class RobustLossFunctions:
    """
    Robust loss functions from robust statistics literature.
    Less sensitive to outliers than standard L2 loss.
    """

    @staticmethod
    def huber_loss(residuals: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        """
        Huber loss: quadratic for small residuals, linear for large.

        L(r) = 0.5 * r^2         if |r| <= delta
               delta * (|r| - 0.5 * delta)  otherwise
        """
        abs_r = torch.abs(residuals)
        quadratic = 0.5 * residuals**2
        linear = delta * (abs_r - 0.5 * delta)

        return torch.where(abs_r <= delta, quadratic, linear)

    @staticmethod
    def tukey_biweight_loss(residuals: torch.Tensor, c: float = 4.685) -> torch.Tensor:
        """
        Tukey's biweight loss: heavily downweights outliers.

        L(r) = (c^2/6) * (1 - (1 - (r/c)^2)^3)  if |r| <= c
               c^2/6                               otherwise
        """
        abs_r = torch.abs(residuals)
        mask = abs_r <= c

        ratio = residuals / c
        biweight = (c**2 / 6) * (1 - (1 - ratio**2) ** 3)
        constant = torch.tensor(c**2 / 6, device=residuals.device)

        return torch.where(mask, biweight, constant)

    @staticmethod
    def cauchy_loss(residuals: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Cauchy loss: robust to extreme outliers.

        L(r) = (c^2 / 2) * log(1 + (r/c)^2)
        """
        return (c**2 / 2) * torch.log(1 + (residuals / c) ** 2)

    @staticmethod
    def geman_mcclure_loss(residuals: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Geman-McClure loss: strongly suppresses outliers.

        L(r) = (r^2 / 2) / (1 + r^2/c^2)
        """
        r_sq = residuals**2
        return (r_sq / 2) / (1 + r_sq / (c**2))

    @staticmethod
    def welsh_loss(residuals: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Welsh loss: exponential decay for large residuals.

        L(r) = (c^2 / 2) * (1 - exp(-r^2/c^2))
        """
        return (c**2 / 2) * (1 - torch.exp(-(residuals**2) / (c**2)))

    @staticmethod
    def fair_loss(residuals: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Fair loss: smooth transition to robustness.

        L(r) = c^2 * (|r|/c - log(1 + |r|/c))
        """
        abs_r = torch.abs(residuals)
        ratio = abs_r / c
        return (c**2) * (ratio - torch.log(1 + ratio))


@register_loss(name="robust_statistical", category="robust")
class RobustStatisticalLoss(BaseLoss):
    """
    Robust Statistical Loss using M-estimators.

    Implements various robust loss functions that are less sensitive to outliers.
    Automatically adapts scale parameter during training.

    Mathematical Formulation:
        L = ρ(r / σ) * σ

    Where:
        - ρ is the robust loss function (Huber, Tukey, Cauchy, etc.)
        - r is the residual (prediction - target)
        - σ is the scale parameter (adaptive)

    Features:
    - Multiple robust loss functions (Huber, Tukey, Cauchy, Geman-McClure)
    - Adaptive scale estimation
    - Outlier detection and handling
    - Automatic scale update

    Example:
        loss = RobustStatisticalLoss(
            robust_type='huber',
            scale=1.0,
            adaptive_scale=True
        )
    """

    def __init__(
        self,
        robust_type: str = "huber",
        scale: float = 1.0,
        adaptive_scale: bool = False,
        scale_update_rate: float = 0.1,
        outlier_threshold: Optional[float] = None,
        reduction: str = "mean",
        device: str = "auto",
    ):
        """
        Initialize robust statistical loss.

        Args:
            robust_type: Type of robust loss ('huber', 'tukey', 'cauchy', 'geman_mcclure')
            scale: Scale parameter for robust loss
            adaptive_scale: Whether to adapt scale during training
            scale_update_rate: Rate for scale updates
            outlier_threshold: Optional threshold for outlier detection
            reduction: Loss reduction method
            device: Device for computation
        """
        super().__init__(reduction=reduction, device=device)

        self.robust_type = robust_type
        self.scale = scale
        self.adaptive_scale = adaptive_scale
        self.scale_update_rate = scale_update_rate
        self.outlier_threshold = outlier_threshold

        # Select loss function
        self.loss_fn = self._get_loss_function()

        # Running statistics for adaptive scale
        self._residual_history = []
        self._outlier_count = 0
        self._total_samples = 0

    def _get_loss_function(self):
        """Get the robust loss function."""
        loss_functions = {
            "huber": lambda r, c: RobustLossFunctions.huber_loss(r, c),
            "tukey": lambda r, c: RobustLossFunctions.tukey_biweight_loss(r, c),
            "cauchy": lambda r, c: RobustLossFunctions.cauchy_loss(r, c),
            "geman_mcclure": lambda r, c: RobustLossFunctions.geman_mcclure_loss(r, c),
            "welsh": lambda r, c: RobustLossFunctions.welsh_loss(r, c),
            "fair": lambda r, c: RobustLossFunctions.fair_loss(r, c),
        }

        if self.robust_type not in loss_functions:
            raise ValueError(f"Unknown robust type: {self.robust_type}")

        return loss_functions[self.robust_type]

    def _update_scale(self, residuals: torch.Tensor) -> None:
        """
        Update scale parameter adaptively using MAD estimator.

        MAD = median(|r - median(r)|) * 1.4826
        """
        if not self.adaptive_scale:
            return

        with torch.no_grad():
            # Compute MAD (Median Absolute Deviation)
            median_r = torch.median(torch.abs(residuals))
            mad = median_r * 1.4826  # Consistent estimator for Gaussian

            # Update scale
            new_scale = max(mad.item(), 1e-6)
            self.scale = (
                1 - self.scale_update_rate
            ) * self.scale + self.scale_update_rate * new_scale

    def _detect_outliers(self, residuals: torch.Tensor) -> torch.Tensor:
        """
        Detect outliers using robust statistics.

        Returns boolean mask of outliers.
        """
        if self.outlier_threshold is None:
            return torch.zeros_like(residuals, dtype=torch.bool)

        # Standardized residuals
        standardized = torch.abs(residuals) / (self.scale + 1e-6)

        # Flag outliers
        outliers = standardized > self.outlier_threshold

        # Update statistics
        self._outlier_count += outliers.sum().item()
        self._total_samples += residuals.numel()

        return outliers

    def _compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Compute robust statistical loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Robust loss
        """
        # Handle classification vs regression
        if predictions.dim() > 1 and predictions.size(-1) > 1 and targets.dim() == 1:
            # Classification: convert targets to one-hot
            num_classes = predictions.size(-1)
            targets_one_hot = torch.zeros_like(predictions)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
            targets = targets_one_hot

        # Compute residuals
        residuals = predictions - targets

        # Detect and count outliers
        outliers = self._detect_outliers(residuals)

        # Scale residuals
        scaled_residuals = residuals / (self.scale + 1e-6)

        # Compute robust loss
        robust_loss = self.loss_fn(scaled_residuals, 1.0)

        # Rescale back
        robust_loss = robust_loss * self.scale

        # Optionally mask outliers
        if self.outlier_threshold is not None:
            robust_loss = torch.where(outliers, robust_loss * 0.5, robust_loss)

        # Update scale adaptively
        if self.adaptive_scale and self.training:
            self._update_scale(residuals.detach())

        # Store for history
        self._residual_history.append(residuals.detach().cpu())
        if len(self._residual_history) > 100:
            self._residual_history.pop(0)

        return robust_loss

    def get_robust_stats(self) -> dict:
        """Get robust loss statistics."""
        stats = {
            "scale": self.scale,
            "robust_type": self.robust_type,
            "adaptive_scale": self.adaptive_scale,
        }

        if self._total_samples > 0:
            stats["outlier_rate"] = self._outlier_count / self._total_samples

        if self._residual_history:
            import numpy as np

            all_residuals = torch.cat(self._residual_history)
            stats["residual_mean"] = all_residuals.mean().item()
            stats["residual_std"] = all_residuals.std().item()
            stats["residual_median"] = all_residuals.median().item()

        return stats

    def reset_statistics(self) -> None:
        """Reset tracking statistics."""
        self._residual_history = []
        self._outlier_count = 0
        self._total_samples = 0

    @classmethod
    def from_config(
        cls, config: RobustStatisticalLossConfig
    ) -> "RobustStatisticalLoss":
        """Create loss from configuration."""
        return cls(
            robust_type=config.robust_type,
            scale=config.scale,
            adaptive_scale=config.adaptive_scale,
            scale_update_rate=config.scale_update_rate,
            outlier_threshold=config.outlier_threshold,
            reduction=config.reduction,
            device=config.device,
        )

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"type={self.robust_type}, "
            f"scale={self.scale:.4f}, "
            f"adaptive={self.adaptive_scale}"
        )


class AdaptiveTrimmedLoss(BaseLoss):
    """
    Adaptive Trimmed Loss for extreme robustness.

    Automatically trims a percentage of largest residuals.
    Combines ideas from trimmed least squares and M-estimators.

    Features:
    - Automatic trimming percentage
    - Adaptive based on outlier contamination
    - Combines with any base loss
    """

    def __init__(
        self,
        base_loss: str = "mse",
        trim_ratio: float = 0.1,
        adaptive_trim: bool = True,
        min_trim: float = 0.0,
        max_trim: float = 0.5,
        reduction: str = "mean",
        device: str = "auto",
    ):
        """
        Initialize adaptive trimmed loss.

        Args:
            base_loss: Base loss function type
            trim_ratio: Initial fraction of samples to trim
            adaptive_trim: Whether to adapt trim ratio
            min_trim: Minimum trim ratio
            max_trim: Maximum trim ratio
            reduction: Loss reduction method
            device: Device for computation
        """
        super().__init__(reduction=reduction, device=device)

        self.base_loss_type = base_loss
        self.trim_ratio = trim_ratio
        self.adaptive_trim = adaptive_trim
        self.min_trim = min_trim
        self.max_trim = max_trim

        # Create base loss
        loss_map = {
            "mse": nn.MSELoss(reduction="none"),
            "l1": nn.L1Loss(reduction="none"),
            "huber": lambda x, y: RobustLossFunctions.huber_loss(x - y),
        }
        self.base_loss_fn = loss_map.get(base_loss, loss_map["mse"])

        # Statistics
        self._trim_history = []

    def _compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute adaptive trimmed loss."""
        # Compute per-sample losses
        if self.base_loss_type == "huber":
            sample_losses = self.base_loss_fn(predictions, targets)
        else:
            sample_losses = self.base_loss_fn(predictions, targets)

        # Determine trim threshold
        if self.adaptive_trim:
            # Estimate contamination using residual distribution
            sorted_losses, _ = torch.sort(sample_losses.flatten())
            n = sorted_losses.numel()

            # Use robust statistics to estimate contamination
            q75_idx = int(0.75 * n)
            q25_idx = int(0.25 * n)
            iqr = sorted_losses[q75_idx] - sorted_losses[q25_idx]

            # Adaptive trim based on distribution tail
            adaptive_threshold = sorted_losses[q75_idx] + 1.5 * iqr

            # Compute effective trim ratio
            trimmed_count = (sample_losses > adaptive_threshold).sum().item()
            effective_trim = trimmed_count / n

            # Clamp to bounds
            effective_trim = max(self.min_trim, min(self.max_trim, effective_trim))
            self.trim_ratio = effective_trim

        # Apply trimming
        k = int(self.trim_ratio * sample_losses.numel())
        if k > 0:
            # Keep only smallest (1 - trim_ratio) losses
            threshold = torch.kthvalue(
                sample_losses.flatten(), sample_losses.numel() - k
            )[0]
            mask = sample_losses <= threshold
            trimmed_losses = sample_losses[mask]
        else:
            trimmed_losses = sample_losses

        # Update history
        self._trim_history.append(self.trim_ratio)

        # Apply reduction
        if self.reduction == "mean":
            return trimmed_losses.mean()
        elif self.reduction == "sum":
            return trimmed_losses.sum()
        else:
            return trimmed_losses

    def get_trim_stats(self) -> dict:
        """Get trimming statistics."""
        if not self._trim_history:
            return {}

        import numpy as np

        return {
            "current_trim_ratio": self.trim_ratio,
            "mean_trim_ratio": np.mean(self._trim_history),
            "max_trim_ratio": np.max(self._trim_history),
            "min_trim_ratio": np.min(self._trim_history),
        }

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"base={self.base_loss_type}, "
            f"trim={self.trim_ratio:.3f}, "
            f"adaptive={self.adaptive_trim}"
        )
