"""
Information-Theoretic Loss
Implements loss functions based on information theory concepts
Uses entropy, mutual information, and KL divergence for learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..core.base_loss import BaseLoss
from ..core.loss_registry import register_loss
from ..config.loss_config import InformationTheoreticLossConfig


class EntropyCalculator:
    """
    Utility class for computing various entropy measures.
    """

    @staticmethod
    def entropy(probs: torch.Tensor, dim: int = -1, eps: float = 1e-10) -> torch.Tensor:
        """
        Compute Shannon entropy.
        H(X) = -sum(p(x) * log(p(x)))
        """
        probs = probs.clamp(min=eps)
        return -(probs * torch.log(probs)).sum(dim=dim)

    @staticmethod
    def conditional_entropy(
        joint_probs: torch.Tensor, eps: float = 1e-10
    ) -> torch.Tensor:
        """
        Compute conditional entropy.
        H(Y|X) = -sum(p(x,y) * log(p(y|x)))
        """
        # joint_probs: (batch, num_x, num_y)
        # Marginal p(x)
        p_x = joint_probs.sum(dim=-1, keepdim=True).clamp(min=eps)

        # Conditional p(y|x)
        p_y_given_x = joint_probs / p_x
        p_y_given_x = p_y_given_x.clamp(min=eps)

        # Conditional entropy
        return -(joint_probs * torch.log(p_y_given_x)).sum(dim=(-2, -1))

    @staticmethod
    def mutual_information(
        probs_x: torch.Tensor,
        probs_y: torch.Tensor,
        joint_probs: torch.Tensor,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """
        Compute mutual information.
        I(X;Y) = sum(p(x,y) * log(p(x,y) / (p(x)*p(y))))
        """
        probs_x = probs_x.clamp(min=eps)
        probs_y = probs_y.clamp(min=eps)
        joint_probs = joint_probs.clamp(min=eps)

        # Product of marginals
        outer = probs_x.unsqueeze(-1) * probs_y.unsqueeze(-2)
        outer = outer.clamp(min=eps)

        # Mutual information
        mi = (joint_probs * torch.log(joint_probs / outer)).sum(dim=(-2, -1))

        return mi

    @staticmethod
    def kl_divergence(
        p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10
    ) -> torch.Tensor:
        """
        Compute KL divergence.
        D_KL(P||Q) = sum(p(x) * log(p(x)/q(x)))
        """
        p = p.clamp(min=eps)
        q = q.clamp(min=eps)
        return (p * torch.log(p / q)).sum(dim=-1)


@register_loss(name="information_theoretic", category="information_theory")
class InformationTheoreticLoss(BaseLoss):
    """
    Information-Theoretic Loss combining multiple information measures.

    Combines cross-entropy with entropy regularization and optional
    mutual information maximization for better representation learning.

    Mathematical Formulation:
        L = L_CE - λ1 * H(predictions) - λ2 * I(predictions; representations)

    Where:
        - L_CE is cross-entropy loss
        - H is entropy (encourages confident predictions)
        - I is mutual information (encourages informative representations)

    Features:
    - Entropy regularization for confident predictions
    - Mutual information maximization
    - KL divergence constraints
    - Temperature scaling

    Example:
        loss = InformationTheoreticLoss(
            entropy_weight=0.1,
            mi_weight=0.05,
            temperature=1.0
        )
    """

    def __init__(
        self,
        use_entropy_regularization: bool = True,
        entropy_weight: float = 0.1,
        use_mutual_information: bool = False,
        mi_weight: float = 0.1,
        use_kl_divergence: bool = False,
        kl_weight: float = 0.1,
        temperature: float = 1.0,
        reduction: str = "mean",
        device: str = "auto",
    ):
        """
        Initialize information-theoretic loss.

        Args:
            use_entropy_regularization: Whether to use entropy regularization
            entropy_weight: Weight for entropy term
            use_mutual_information: Whether to use MI term
            mi_weight: Weight for MI term
            use_kl_divergence: Whether to use KL divergence term
            kl_weight: Weight for KL term
            temperature: Temperature for softmax sharpening
            reduction: Loss reduction method
            device: Device for computation
        """
        super().__init__(reduction=reduction, device=device)

        self.use_entropy_regularization = use_entropy_regularization
        self.entropy_weight = entropy_weight
        self.use_mutual_information = use_mutual_information
        self.mi_weight = mi_weight
        self.use_kl_divergence = use_kl_divergence
        self.kl_weight = kl_weight
        self.temperature = temperature

        # Prior distribution (uniform)
        self.register_buffer("uniform_prior", None)

    def _preprocess_inputs(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply temperature scaling and compute probabilities."""
        # Ensure targets are on same device as predictions
        targets = targets.to(predictions.device)

        # Temperature scaling
        scaled_preds = predictions / self.temperature

        # Compute probabilities
        self._probs = F.softmax(scaled_preds, dim=-1)
        self._log_probs = F.log_softmax(scaled_preds, dim=-1)

        return scaled_preds, targets

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute information-theoretic loss.

        Args:
            predictions: Model predictions (logits)
            targets: Ground truth labels

        Returns:
            Combined information-theoretic loss
        """
        batch_size = predictions.size(0)

        # 1. Cross-entropy loss
        ce_loss = F.nll_loss(self._log_probs, targets, reduction=self.reduction)

        total_loss = ce_loss

        # 2. Entropy regularization (encourage confident predictions)
        if self.use_entropy_regularization:
            entropy = EntropyCalculator.entropy(self._probs, dim=-1)

            if self.reduction == "mean":
                entropy_reg = -entropy.mean()  # Negative because we want to maximize
            elif self.reduction == "sum":
                entropy_reg = -entropy.sum()
            else:
                entropy_reg = -entropy

            total_loss = total_loss + self.entropy_weight * entropy_reg

        # 3. Mutual information with representations
        if self.use_mutual_information:
            # Compute marginal probabilities
            pred_marginal = self._probs.mean(dim=0)

            # Compute joint distribution (approximated)
            # This is a simplified version - in practice, you'd use representations
            joint = self._probs.unsqueeze(1) * pred_marginal.unsqueeze(
                0
            )  # (batch, classes, classes)

            # Mutual information
            mi = EntropyCalculator.mutual_information(
                self._probs, pred_marginal.expand(batch_size, -1), joint
            )

            if self.reduction == "mean":
                mi_loss = -mi.mean()  # Maximize MI
            elif self.reduction == "sum":
                mi_loss = -mi.sum()
            else:
                mi_loss = -mi

            total_loss = total_loss + self.mi_weight * mi_loss

        # 4. KL divergence from uniform prior
        if self.use_kl_divergence:
            num_classes = predictions.size(-1)

            if self.uniform_prior is None or self.uniform_prior.size(0) != num_classes:
                self.uniform_prior = (
                    torch.ones(num_classes, device=predictions.device) / num_classes
                )

            # Ensure uniform_prior is on the same device as predictions
            if self.uniform_prior.device != predictions.device:
                self.uniform_prior = self.uniform_prior.to(predictions.device)

            # KL divergence for each prediction
            kl_div = EntropyCalculator.kl_divergence(self._probs, self.uniform_prior)

            if self.reduction == "mean":
                kl_loss = kl_div.mean()
            elif self.reduction == "sum":
                kl_loss = kl_div.sum()
            else:
                kl_loss = kl_div

            total_loss = total_loss + self.kl_weight * kl_loss

        return total_loss

    def get_information_stats(self) -> dict:
        """Get information-theoretic statistics."""
        return {
            "entropy_weight": self.entropy_weight,
            "mi_weight": self.mi_weight,
            "kl_weight": self.kl_weight,
            "temperature": self.temperature,
        }

    @classmethod
    def from_config(
        cls, config: InformationTheoreticLossConfig
    ) -> "InformationTheoreticLoss":
        """Create loss from configuration."""
        return cls(
            use_entropy_regularization=config.use_entropy_regularization,
            entropy_weight=config.entropy_weight,
            use_mutual_information=config.use_mutual_information,
            mi_weight=config.mi_weight,
            use_kl_divergence=config.use_kl_divergence,
            kl_weight=config.kl_weight,
            temperature=config.temperature,
            reduction=config.reduction,
            device=config.device,
        )

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"entropy={self.use_entropy_regularization}, "
            f"mi={self.use_mutual_information}, "
            f"kl={self.use_kl_divergence}, "
            f"temp={self.temperature}"
        )


class VariationalInformationLoss(BaseLoss):
    """
    Variational Information Maximization Loss.

    Uses variational bounds to estimate and maximize mutual information.
    Based on the InfoNCE bound and related methods.

    Features:
    - InfoNCE contrastive loss
    - Variational lower bounds
    - Noise-contrastive estimation
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_negatives: int = 100,
        temperature: float = 0.07,
        reduction: str = "mean",
        device: str = "auto",
    ):
        """
        Initialize variational information loss.

        Args:
            embedding_dim: Dimension of embeddings
            num_negatives: Number of negative samples
            temperature: Temperature for softmax
            reduction: Loss reduction method
            device: Device for computation
        """
        super().__init__(reduction=reduction, device=device)

        self.embedding_dim = embedding_dim
        self.num_negatives = num_negatives
        self.temperature = temperature

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def _compute_loss(
        self, representations: torch.Tensor, positive_pairs: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            representations: Input representations (batch, dim)
            positive_pairs: Positive pair indices (batch,)

        Returns:
            InfoNCE loss
        """
        batch_size = representations.size(0)

        # Project representations
        z_i = F.normalize(self.projection(representations), dim=-1)
        z_j = F.normalize(self.projection(representations[positive_pairs]), dim=-1)

        # Compute similarity scores
        sim_pos = (z_i * z_j).sum(dim=-1) / self.temperature

        # Compute similarities with all negatives
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature

        # InfoNCE loss
        logits = torch.cat([sim_pos.unsqueeze(1), sim_matrix], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        return loss

    def extra_repr(self) -> str:
        """String representation."""
        return f"dim={self.embedding_dim}, temp={self.temperature}"
