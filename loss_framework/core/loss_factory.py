"""
Loss Factory Module
Implements Factory pattern for creating loss function instances
Provides flexible object creation with configuration support
"""

import torch
from typing import Optional, Dict, Any, Union
from .loss_registry import LossRegistry
from ..config.loss_config import LossConfig


class LossFactory:
    """
    Factory pattern for creating loss function instances.

    Encapsulates object creation logic and supports:
    - Creating losses from configuration objects
    - Creating losses from dictionaries
    - Creating standard PyTorch losses
    - Creating custom registered losses

    Usage:
        # From config
        config = LossConfig(loss_type='cross_entropy')
        loss = LossFactory.create_from_config(config)

        # From dict
        loss = LossFactory.create_from_dict({'loss_type': 'mse', 'reduction': 'sum'})

        # Standard losses
        loss = LossFactory.create_standard('cross_entropy')
    """

    @staticmethod
    def create_from_config(config: LossConfig) -> torch.nn.Module:
        """
        Create loss function from configuration object.

        Args:
            config: LossConfig instance

        Returns:
            Configured loss function instance
        """
        loss_type = config.loss_type.lower()

        # First check if it's a registered custom loss
        registered_loss = LossRegistry.get(loss_type)
        if registered_loss is not None:
            return registered_loss(reduction=config.reduction, **config.hyperparameters)

        # Otherwise create standard PyTorch loss
        return LossFactory.create_standard(
            loss_type=loss_type,
            reduction=config.reduction,
            weight=config.class_weights,
            **config.hyperparameters,
        )

    @staticmethod
    def create_from_dict(config_dict: Dict[str, Any]) -> torch.nn.Module:
        """
        Create loss function from dictionary configuration.

        Args:
            config_dict: Dictionary with configuration parameters

        Returns:
            Configured loss function instance
        """
        # Create config object from dict
        config = LossConfig.from_dict(config_dict)
        return LossFactory.create_from_config(config)

    @staticmethod
    def create_standard(
        loss_type: str,
        reduction: str = "mean",
        weight: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.nn.Module:
        """
        Create standard PyTorch loss function.

        Args:
            loss_type: Type of loss ('mse', 'cross_entropy', 'l1', etc.)
            reduction: Reduction method
            weight: Optional class weights
            **kwargs: Additional loss-specific parameters

        Returns:
            Standard PyTorch loss instance
        """
        loss_type = loss_type.lower()

        # Classification losses
        if loss_type in ["cross_entropy", "crossentropy", "ce"]:
            return torch.nn.CrossEntropyLoss(
                weight=weight,
                reduction=reduction,
                ignore_index=kwargs.get("ignore_index", -100),
                label_smoothing=kwargs.get("label_smoothing", 0.0),
            )

        elif loss_type in ["bce", "binary_cross_entropy", "bceloss"]:
            return torch.nn.BCELoss(weight=weight, reduction=reduction)

        elif loss_type in ["bce_with_logits", "bcewithlogits"]:
            return torch.nn.BCEWithLogitsLoss(
                weight=weight, reduction=reduction, pos_weight=kwargs.get("pos_weight")
            )

        elif loss_type in ["nll", "nllloss", "negative_log_likelihood"]:
            return torch.nn.NLLLoss(
                weight=weight,
                reduction=reduction,
                ignore_index=kwargs.get("ignore_index", -100),
            )

        # Regression losses
        elif loss_type in ["mse", "mean_squared_error", "l2"]:
            return torch.nn.MSELoss(reduction=reduction)

        elif loss_type in ["l1", "mae", "mean_absolute_error"]:
            return torch.nn.L1Loss(reduction=reduction)

        elif loss_type in ["smooth_l1", "smoothl1", "huber"]:
            return torch.nn.SmoothL1Loss(
                reduction=reduction, beta=kwargs.get("beta", 1.0)
            )

        # Probabilistic losses
        elif loss_type in ["kl_div", "kldiv", "kullback_leibler"]:
            return torch.nn.KLDivLoss(
                reduction=reduction, log_target=kwargs.get("log_target", False)
            )

        # Ranking losses
        elif loss_type in ["margin_ranking", "marginranking"]:
            return torch.nn.MarginRankingLoss(
                margin=kwargs.get("margin", 0.0), reduction=reduction
            )

        elif loss_type in ["triplet_margin", "tripletmargin"]:
            return torch.nn.TripletMarginLoss(
                margin=kwargs.get("margin", 1.0),
                p=kwargs.get("p", 2),
                reduction=reduction,
            )

        # Cosine losses
        elif loss_type in ["cosine_embedding", "cosineembedding"]:
            return torch.nn.CosineEmbeddingLoss(
                margin=kwargs.get("margin", 0.0), reduction=reduction
            )

        # Multi-label losses
        elif loss_type in ["multi_label_margin", "multilabelmargin"]:
            return torch.nn.MultiLabelMarginLoss(reduction=reduction)

        elif loss_type in ["multi_label_soft_margin", "multilabelsoftmargin"]:
            return torch.nn.MultiLabelSoftMarginLoss(weight=weight, reduction=reduction)

        else:
            raise ValueError(
                f"Unknown standard loss type: {loss_type}. "
                f"Available types: mse, l1, cross_entropy, bce, smooth_l1, "
                f"kl_div, margin_ranking, triplet_margin, cosine_embedding"
            )

    @staticmethod
    def create_composite(
        losses: Dict[str, Union[torch.nn.Module, Dict[str, Any]]],
        weights: Optional[Dict[str, float]] = None,
    ) -> "CompositeLoss":
        """
        Create a composite loss from multiple losses.

        Args:
            losses: Dictionary of loss name to loss instance or config dict
            weights: Optional weights for each loss

        Returns:
            CompositeLoss instance
        """
        from .composite_loss import CompositeLoss

        # Convert dict configs to loss instances
        loss_instances = {}
        for name, loss_def in losses.items():
            if isinstance(loss_def, dict):
                loss_instances[name] = LossFactory.create_from_dict(loss_def)
            elif isinstance(loss_def, torch.nn.Module):
                loss_instances[name] = loss_def
            else:
                raise TypeError(f"Invalid loss definition for {name}: {type(loss_def)}")

        return CompositeLoss(loss_instances, weights)

    @staticmethod
    def list_available_losses() -> Dict[str, list]:
        """List all available loss types."""
        standard_losses = [
            "mse",
            "l1",
            "cross_entropy",
            "bce",
            "bce_with_logits",
            "nll",
            "smooth_l1",
            "kl_div",
            "margin_ranking",
            "triplet_margin",
            "cosine_embedding",
            "multi_label_margin",
            "multi_label_soft_margin",
        ]

        registered_losses = LossRegistry.list_losses()

        return {
            "standard": standard_losses,
            "registered": registered_losses,
            "all": standard_losses + registered_losses,
        }
