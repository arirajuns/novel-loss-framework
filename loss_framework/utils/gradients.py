"""
Gradient Utilities Module
Provides tools for gradient analysis and manipulation
"""

import torch
from typing import Dict, Optional, Tuple, List
import numpy as np


class GradientUtils:
    """
    Utility class for gradient-related operations.
    Provides tools for analysis, visualization, and manipulation of gradients.
    """

    @staticmethod
    def compute_gradient_norm(parameters, norm_type: float = 2.0) -> float:
        """
        Compute the gradient norm across all parameters.

        Args:
            parameters: Model parameters
            norm_type: Type of norm (2.0 for L2, 1.0 for L1, etc.)

        Returns:
            Gradient norm value
        """
        if isinstance(parameters, torch.nn.Module):
            parameters = parameters.parameters()

        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type

        total_norm = total_norm ** (1.0 / norm_type)
        return total_norm

    @staticmethod
    def clip_gradients(parameters, max_norm: float, norm_type: float = 2.0) -> float:
        """
        Clip gradients by norm.

        Args:
            parameters: Model parameters
            max_norm: Maximum gradient norm
            norm_type: Type of norm

        Returns:
            Total gradient norm before clipping
        """
        if isinstance(parameters, torch.nn.Module):
            parameters = parameters.parameters()

        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)

    @staticmethod
    def get_gradient_stats(parameters) -> Dict[str, float]:
        """
        Get comprehensive gradient statistics.

        Args:
            parameters: Model parameters

        Returns:
            Dictionary of gradient statistics
        """
        if isinstance(parameters, torch.nn.Module):
            parameters = parameters.parameters()

        grads = []
        for p in parameters:
            if p.grad is not None:
                grads.append(p.grad.data.flatten())

        if len(grads) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "norm": 0.0}

        all_grads = torch.cat(grads)

        return {
            "mean": all_grads.mean().item(),
            "std": all_grads.std().item(),
            "min": all_grads.min().item(),
            "max": all_grads.max().item(),
            "norm": all_grads.norm().item(),
        }

    @staticmethod
    def check_gradient_health(parameters) -> Dict[str, any]:
        """
        Check for common gradient issues.

        Args:
            parameters: Model parameters

        Returns:
            Dictionary with health check results
        """
        if isinstance(parameters, torch.nn.Module):
            parameters = parameters.parameters()

        issues = []
        stats = {
            "total_params": 0,
            "params_with_grad": 0,
            "params_without_grad": 0,
            "vanishing_grads": 0,
            "exploding_grads": 0,
            "nan_grads": 0,
            "inf_grads": 0,
        }

        for p in parameters:
            stats["total_params"] += 1

            if p.grad is None:
                stats["params_without_grad"] += 1
                continue

            stats["params_with_grad"] += 1
            grad_norm = p.grad.data.norm().item()

            # Check for NaN
            if torch.isnan(p.grad).any():
                stats["nan_grads"] += 1
                issues.append(f"Parameter has NaN gradients")

            # Check for Inf
            if torch.isinf(p.grad).any():
                stats["inf_grads"] += 1
                issues.append(f"Parameter has Inf gradients")

            # Check for vanishing gradients
            if grad_norm < 1e-7:
                stats["vanishing_grads"] += 1

            # Check for exploding gradients
            if grad_norm > 1e3:
                stats["exploding_grads"] += 1

        stats["has_issues"] = len(issues) > 0
        stats["issues"] = issues

        return stats

    @staticmethod
    def compute_hessian_diagonal(
        loss: torch.Tensor, parameters: List[torch.Tensor], damping: float = 1e-3
    ) -> List[torch.Tensor]:
        """
        Compute diagonal of Hessian matrix (second derivatives).

        Args:
            loss: Scalar loss tensor
            parameters: Parameters to compute Hessian for
            damping: Damping factor for numerical stability

        Returns:
            List of diagonal Hessian elements for each parameter
        """
        hessian_diag = []

        # Compute first derivatives
        grads = torch.autograd.grad(
            loss, parameters, create_graph=True, retain_graph=True
        )

        # Compute second derivatives (diagonal elements)
        for grad, param in zip(grads, parameters):
            if grad is not None:
                # Compute gradient of gradient (diagonal of Hessian)
                diag = torch.autograd.grad(
                    grad.sum(), param, retain_graph=True, allow_unused=True
                )[0]

                if diag is not None:
                    hessian_diag.append(diag + damping)
                else:
                    hessian_diag.append(torch.zeros_like(param))
            else:
                hessian_diag.append(torch.zeros_like(param))

        return hessian_diag

    @staticmethod
    def gradient_penalty(
        discriminator: torch.nn.Module,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        lambda_gp: float = 10.0,
    ) -> torch.Tensor:
        """
        Compute gradient penalty (used in WGAN-GP).

        Args:
            discriminator: Discriminator network
            real_data: Real data samples
            fake_data: Generated data samples
            lambda_gp: Gradient penalty coefficient

        Returns:
            Gradient penalty loss
        """
        batch_size = real_data.size(0)

        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=real_data.device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)

        # Compute discriminator output
        d_interpolates = discriminator(interpolates)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean() * lambda_gp

        return penalty

    @staticmethod
    def get_layer_gradients(
        model: torch.nn.Module, layer_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract gradients from specific layers.

        Args:
            model: Neural network model
            layer_names: Names of layers to extract (None for all with gradients)

        Returns:
            Dictionary mapping layer names to gradients
        """
        gradients = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                if layer_names is None or name in layer_names:
                    gradients[name] = param.grad.data.clone()

        return gradients

    @staticmethod
    def visualize_gradient_flow(model: torch.nn.Module) -> Dict[str, float]:
        """
        Compute gradient flow statistics per layer.

        Args:
            model: Neural network model

        Returns:
            Dictionary with per-layer gradient norms
        """
        layer_grads = {}

        for name, param in model.named_parameters():
            if param.grad is not None and "weight" in name:
                layer_name = name.replace(".weight", "")
                grad_norm = param.grad.data.norm().item()
                layer_grads[layer_name] = grad_norm

        return layer_grads
