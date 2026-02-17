"""
Geometric Distance Loss
Implements loss functions based on Riemannian geometry and manifold learning
Uses geometric concepts for better feature representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from ..core.base_loss import BaseLoss
from ..core.loss_registry import register_loss
from ..config.loss_config import GeometricLossConfig


class ManifoldGeometry:
    """
    Geometric operations on different manifolds.
    Supports Euclidean, Spherical, and Hyperbolic geometries.
    """

    @staticmethod
    def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean distance."""
        return torch.norm(x - y, dim=-1)

    @staticmethod
    def spherical_distance(
        x: torch.Tensor, y: torch.Tensor, curvature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute geodesic distance on sphere.

        Formula: d(x, y) = arccos(<x, y> / (||x|| ||y||)) / sqrt(curvature)
        """
        # Normalize to unit sphere
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)

        # Compute cosine similarity
        cos_sim = (x_norm * y_norm).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)

        # Compute spherical distance
        distance = torch.acos(cos_sim) / math.sqrt(curvature)

        return distance

    @staticmethod
    def hyperbolic_distance(
        x: torch.Tensor, y: torch.Tensor, curvature: float = 1.0, eps: float = 1e-7
    ) -> torch.Tensor:
        """
        Compute distance in hyperbolic space (Poincaré ball model).

        Formula: d(x, y) = arccosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
        
        Fixed: Added robust numerical handling for boundary cases.
        """
        # Compute norms with numerical stability
        x_norm_sq = (x ** 2).sum(dim=-1)
        y_norm_sq = (y ** 2).sum(dim=-1)

        # Clamp norms to valid range (inside Poincaré ball)
        # Use smaller epsilon for more stable boundaries
        boundary_eps = 1e-5
        x_norm_sq = torch.clamp(x_norm_sq, max=1 - boundary_eps)
        y_norm_sq = torch.clamp(y_norm_sq, max=1 - boundary_eps)

        # Compute squared difference
        diff_norm_sq = ((x - y) ** 2).sum(dim=-1)

        # Compute denominator with additional numerical safety
        denom = (1 - x_norm_sq) * (1 - y_norm_sq)
        denom = torch.clamp(denom, min=eps)

        # Compute argument for arccosh with safety bounds
        arg = 1 + 2 * diff_norm_sq / denom
        # Clamp to valid range for arccosh (>= 1)
        arg = torch.clamp(arg, min=1.0 + eps, max=1e6)

        # Compute arccosh with numerical stability
        # arccosh(x) = log(x + sqrt(x^2 - 1))
        distance = torch.acosh(arg) / math.sqrt(curvature)

        # Handle any NaN or Inf values
        distance = torch.where(torch.isfinite(distance), distance, torch.zeros_like(distance))

        return distance

    @staticmethod
    def project_to_manifold(
        x: torch.Tensor, manifold_type: str, curvature: float = 1.0
    ) -> torch.Tensor:
        """Project points onto manifold."""
        if manifold_type == "euclidean":
            return x
        elif manifold_type == "spherical":
            return F.normalize(x, p=2, dim=-1) / math.sqrt(curvature)
        elif manifold_type == "hyperbolic":
            # Project to Poincaré ball
            norm = torch.norm(x, dim=-1, keepdim=True)
            return x / (norm + 1e-7) * torch.tanh(norm) * (1 - 1e-7)
        else:
            raise ValueError(f"Unknown manifold type: {manifold_type}")


@register_loss(name="geometric", category="geometric")
class GeometricDistanceLoss(BaseLoss):
    """
    Geometric Distance Loss on Riemannian manifolds.

    Computes loss based on geodesic distances on different manifolds.
    Supports Euclidean, Spherical, and Hyperbolic geometries.

    Mathematical Formulation:
        L = d_M(predictions, targets)

    Where d_M is the geodesic distance on manifold M.

    Features:
    - Multiple manifold geometries
    - Automatic projection to manifold
    - Chordal vs geodesic distance options
    - Differentiable operations

    Example:
        loss = GeometricDistanceLoss(
            manifold_type='spherical',
            distance_metric='geodesic',
            curvature=1.0
        )
    """

    def __init__(
        self,
        manifold_type: str = "euclidean",
        distance_metric: str = "geodesic",
        curvature: float = 1.0,
        embedding_dim: int = 128,
        project_to_manifold: bool = True,
        reduction: str = "mean",
        device: str = "auto",
    ):
        """
        Initialize geometric distance loss.

        Args:
            manifold_type: Type of manifold ('euclidean', 'spherical', 'hyperbolic')
            distance_metric: Distance metric ('geodesic', 'chordal')
            curvature: Curvature parameter for manifold
            embedding_dim: Dimension of embedding space
            project_to_manifold: Whether to project inputs to manifold
            reduction: Loss reduction method
            device: Device for computation
        """
        super().__init__(reduction=reduction, device=device)

        self.manifold_type = manifold_type
        self.distance_metric = distance_metric
        self.curvature = curvature
        self.embedding_dim = embedding_dim
        self.project_to_manifold = project_to_manifold

        # Select distance function
        self.distance_fn = self._get_distance_function()

        # Embedding projection layer
        self.embedding_proj = nn.Linear(embedding_dim, embedding_dim).to(self.device)

    def _get_distance_function(self):
        """Get the appropriate distance function."""
        if self.distance_metric == "geodesic":
            if self.manifold_type == "euclidean":
                return ManifoldGeometry.euclidean_distance
            elif self.manifold_type == "spherical":
                return lambda x, y: ManifoldGeometry.spherical_distance(
                    x, y, self.curvature
                )
            elif self.manifold_type == "hyperbolic":
                return lambda x, y: ManifoldGeometry.hyperbolic_distance(
                    x, y, self.curvature
                )
        elif self.distance_metric == "chordal":
            # Chordal distance (Euclidean in embedding space)
            return ManifoldGeometry.euclidean_distance
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _preprocess_inputs(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project inputs to embedding space and manifold."""
        # Project to embedding space
        pred_embed = self.embedding_proj(predictions)
        target_embed = self.embedding_proj(targets)

        if self.project_to_manifold:
            # Project to manifold
            pred_embed = ManifoldGeometry.project_to_manifold(
                pred_embed, self.manifold_type, self.curvature
            )
            target_embed = ManifoldGeometry.project_to_manifold(
                target_embed, self.manifold_type, self.curvature
            )

        return pred_embed, target_embed

    def _compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute geometric distance loss."""
        # Compute pairwise distances
        distances = self.distance_fn(predictions, targets)

        return distances

    def get_manifold_info(self) -> dict:
        """Get manifold information."""
        return {
            "type": self.manifold_type,
            "curvature": self.curvature,
            "dimension": self.embedding_dim,
            "distance_metric": self.distance_metric,
        }

    @classmethod
    def from_config(cls, config: GeometricLossConfig) -> "GeometricDistanceLoss":
        """Create loss from configuration."""
        return cls(
            manifold_type=config.manifold_type,
            distance_metric=config.distance_metric,
            curvature=config.curvature,
            embedding_dim=config.embedding_dim,
            project_to_manifold=config.project_to_manifold,
            reduction=config.reduction,
            device=config.device,
        )

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"manifold={self.manifold_type}, "
            f"curvature={self.curvature}, "
            f"metric={self.distance_metric}"
        )


class HyperbolicEmbeddingLoss(BaseLoss):
    """
    Hyperbolic Embedding Loss for hierarchical data.

    Specifically designed for tree-structured or hierarchical data.
    Uses hyperbolic geometry to better represent hierarchical relationships.

    Features:
    - Hierarchical distance preservation
    - Parent-child relationship modeling
    - Learnable hyperbolic embeddings
    """

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 128,
        curvature: float = 1.0,
        lambda_structure: float = 1.0,
        lambda_smooth: float = 0.1,
        reduction: str = "mean",
        device: str = "auto",
    ):
        """
        Initialize hyperbolic embedding loss.

        Args:
            num_nodes: Number of nodes in hierarchy
            embedding_dim: Dimension of hyperbolic embeddings
            curvature: Hyperbolic curvature
            lambda_structure: Weight for structure preservation
            lambda_smooth: Weight for smoothness regularization
            reduction: Loss reduction method
            device: Device for computation
        """
        super().__init__(reduction=reduction, device=device)

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.curvature = curvature
        self.lambda_structure = lambda_structure
        self.lambda_smooth = lambda_smooth

        # Learnable embeddings
        self.embeddings = nn.Parameter(torch.randn(num_nodes, embedding_dim) * 0.01)

    def _compute_loss(
        self, node_pairs: torch.Tensor, distances: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Compute hyperbolic embedding loss.

        Args:
            node_pairs: Tensor of shape (N, 2) with node indices
            distances: True distances between node pairs

        Returns:
            Hyperbolic embedding loss
        """
        # Get embeddings for node pairs
        emb_i = self.embeddings[node_pairs[:, 0]]
        emb_j = self.embeddings[node_pairs[:, 1]]

        # Project to Poincaré ball
        emb_i = ManifoldGeometry.project_to_manifold(
            emb_i, "hyperbolic", self.curvature
        )
        emb_j = ManifoldGeometry.project_to_manifold(
            emb_j, "hyperbolic", self.curvature
        )

        # Compute hyperbolic distances
        hyp_distances = ManifoldGeometry.hyperbolic_distance(
            emb_i, emb_j, self.curvature
        )

        # Structure preservation loss
        structure_loss = F.mse_loss(hyp_distances, distances)

        # Smoothness regularization (keep embeddings small)
        smooth_loss = torch.norm(self.embeddings, p=2, dim=1).mean()

        # Combined loss
        total_loss = (
            self.lambda_structure * structure_loss + self.lambda_smooth * smooth_loss
        )

        return total_loss

    def get_embeddings(self) -> torch.Tensor:
        """Get current embeddings."""
        return ManifoldGeometry.project_to_manifold(
            self.embeddings, "hyperbolic", self.curvature
        )
