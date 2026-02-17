"""
Metrics Calculator Module
Provides evaluation metrics for loss functions and model performance
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class MetricsCalculator:
    """
    Comprehensive metrics calculation for model evaluation.
    Supports both classification and regression metrics.
    """

    @staticmethod
    def classification_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: Optional[int] = None,
        average: str = "macro",
        compute_confusion: bool = True,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.

        Args:
            predictions: Predicted logits or probabilities (N, C) or class labels (N,)
            targets: Ground truth labels (N,)
            num_classes: Number of classes (inferred if not provided)
            average: Averaging method for multi-class ('macro', 'micro', 'weighted')
            compute_confusion: Whether to compute confusion matrix

        Returns:
            Dictionary of metrics
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Infer num_classes if not provided
        if num_classes is None:
            num_classes = max(predictions.max(), targets.max()) + 1

        # Get predicted classes
        if predictions.ndim > 1:
            # Probabilities/logits -> class predictions
            pred_classes = predictions.argmax(axis=1)
            probs = torch.softmax(torch.from_numpy(predictions), dim=1).numpy()
        else:
            pred_classes = predictions
            probs = None

        metrics = {
            "accuracy": accuracy_score(targets, pred_classes),
            "precision": precision_score(
                targets, pred_classes, average=average, zero_division=0
            ),
            "recall": recall_score(
                targets, pred_classes, average=average, zero_division=0
            ),
            "f1_score": f1_score(
                targets, pred_classes, average=average, zero_division=0
            ),
        }

        # Per-class metrics
        precision_per_class = precision_score(
            targets, pred_classes, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            targets, pred_classes, average=None, zero_division=0
        )

        for i in range(min(num_classes, len(precision_per_class))):
            metrics[f"precision_class_{i}"] = precision_per_class[i]
            metrics[f"recall_class_{i}"] = recall_per_class[i]

        # Confusion matrix
        if compute_confusion:
            cm = confusion_matrix(targets, pred_classes, labels=range(num_classes))
            metrics["confusion_matrix"] = cm

        # ROC-AUC (for binary or one-vs-rest)
        if probs is not None and num_classes == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(targets, probs[:, 1])
                metrics["average_precision"] = average_precision_score(
                    targets, probs[:, 1]
                )
            except ValueError:
                pass

        return metrics

    @staticmethod
    def regression_metrics(
        predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.

        Args:
            predictions: Predicted values
            targets: Ground truth values

        Returns:
            Dictionary of metrics
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Flatten if needed
        predictions = predictions.flatten()
        targets = targets.flatten()

        mse = mean_squared_error(targets, predictions)

        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(targets, predictions),
            "r2": r2_score(targets, predictions),
            "mape": np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100,
            "max_error": np.max(np.abs(targets - predictions)),
        }

    @staticmethod
    def loss_landscape_metrics(
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        direction: Optional[torch.Tensor] = None,
        num_points: int = 50,
        distance: float = 1.0,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Analyze loss landscape along a direction.

        Args:
            model: Neural network model
            loss_fn: Loss function
            data_loader: Data loader for evaluation
            direction: Direction vector (None for random direction)
            num_points: Number of points to evaluate
            distance: Distance to explore in each direction

        Returns:
            Dictionary with loss landscape data
        """
        # Get current parameters
        original_params = [p.clone() for p in model.parameters()]

        # Create random direction if not provided
        if direction is None:
            direction = []
            for p in model.parameters():
                direction.append(torch.randn_like(p))
            # Normalize
            norm = np.sqrt(sum([d.norm().item() ** 2 for d in direction]))
            direction = [d / norm for d in direction]

        # Evaluate loss at different points
        alphas = np.linspace(-distance, distance, num_points)
        losses = []

        for alpha in alphas:
            # Update parameters
            for p, d in zip(model.parameters(), direction):
                p.data = original_params[alpha] + alpha * d

            # Compute loss
            total_loss = 0.0
            count = 0
            with torch.no_grad():
                for batch_data, batch_targets in data_loader:
                    predictions = model(batch_data)
                    loss = loss_fn(predictions, batch_targets)
                    total_loss += loss.item()
                    count += 1

            losses.append(total_loss / count)

        # Restore original parameters
        for p, orig in zip(model.parameters(), original_params):
            p.data = orig

        losses = np.array(losses)

        return {
            "alphas": alphas,
            "losses": losses,
            "min_loss": losses.min(),
            "max_loss": losses.max(),
            "loss_range": losses.max() - losses.min(),
            "smoothness": np.std(np.diff(losses)),
        }

    @staticmethod
    def compute_calibration(
        predictions: torch.Tensor, targets: torch.Tensor, num_bins: int = 10
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Compute calibration metrics (Expected Calibration Error).

        Args:
            predictions: Predicted probabilities (N, C)
            targets: Ground truth labels (N,)
            num_bins: Number of bins for calibration

        Returns:
            Dictionary with calibration metrics
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Get predicted probabilities and classes
        pred_probs = predictions.max(axis=1)
        pred_classes = predictions.argmax(axis=1)
        accuracies = (pred_classes == targets).astype(float)

        # Create bins
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        bin_accs = []
        bin_confs = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in bin
            in_bin = (pred_probs > bin_lower) & (pred_probs <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = pred_probs[in_bin].mean()

                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                bin_accs.append(accuracy_in_bin)
                bin_confs.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_accs.append(0)
                bin_confs.append(0)
                bin_counts.append(0)

        return {
            "ece": ece,
            "bin_accuracies": np.array(bin_accs),
            "bin_confidences": np.array(bin_confs),
            "bin_counts": np.array(bin_counts),
            "bin_boundaries": bin_boundaries,
        }

    @staticmethod
    def convergence_metrics(
        loss_history: List[float], window_size: int = 10
    ) -> Dict[str, float]:
        """
        Analyze convergence properties from loss history.

        Args:
            loss_history: List of loss values over training
            window_size: Window size for smoothing

        Returns:
            Dictionary with convergence metrics
        """
        if len(loss_history) < 2:
            return {
                "final_loss": loss_history[-1] if loss_history else 0,
                "converged": False,
            }

        loss_array = np.array(loss_history)

        # Smooth loss
        if len(loss_array) >= window_size:
            smoothed = np.convolve(
                loss_array, np.ones(window_size) / window_size, mode="valid"
            )
        else:
            smoothed = loss_array

        # Compute metrics
        initial_loss = loss_array[0]
        final_loss = loss_array[-1]

        # Check convergence (loss decreasing)
        recent_losses = (
            loss_array[-window_size:] if len(loss_array) >= window_size else loss_array
        )
        converged = np.std(recent_losses) < 0.01 * np.abs(final_loss)

        # Plateau detection
        if len(smoothed) >= window_size:
            recent_slope = (smoothed[-1] - smoothed[-window_size]) / window_size
            plateau = abs(recent_slope) < 1e-5
        else:
            plateau = False

        return {
            "initial_loss": float(initial_loss),
            "final_loss": float(final_loss),
            "improvement": float(initial_loss - final_loss),
            "improvement_ratio": float((initial_loss - final_loss) / initial_loss)
            if initial_loss != 0
            else 0,
            "converged": converged,
            "plateau": plateau,
            "min_loss": float(loss_array.min()),
            "max_loss": float(loss_array.max()),
            "loss_std": float(loss_array.std()),
        }
