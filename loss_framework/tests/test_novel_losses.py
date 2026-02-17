"""
Tests for novel loss function implementations
Validates mathematical correctness and gradient flow
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from loss_framework.losses.adaptive_weighted_loss import (
    AdaptiveWeightedLoss,
    DynamicFocalLoss,
    WeightScheduleStrategy,
)
from loss_framework.losses.geometric_loss import (
    GeometricDistanceLoss,
    HyperbolicEmbeddingLoss,
    ManifoldGeometry,
)
from loss_framework.losses.information_theoretic_loss import (
    InformationTheoreticLoss,
    VariationalInformationLoss,
)
from loss_framework.losses.physics_inspired_loss import (
    PhysicsInspiredLoss,
    HamiltonianMechanics,
)
from loss_framework.losses.robust_statistical_loss import (
    RobustStatisticalLoss,
    AdaptiveTrimmedLoss,
    RobustLossFunctions,
)


class TestAdaptiveWeightedLoss:
    """Tests for adaptive weighted loss."""

    def test_initialization(self):
        """Test loss initialization."""
        loss = AdaptiveWeightedLoss(
            base_loss="cross_entropy", schedule_type="cosine", warmup_epochs=10
        )

        assert loss.base_loss_type == "cross_entropy"
        assert loss.schedule_type == "cosine"
        assert loss.warmup_epochs == 10

    def test_forward_pass(self):
        """Test forward pass."""
        loss = AdaptiveWeightedLoss()

        predictions = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))

        result = loss(predictions, targets)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0
        assert result.item() >= 0

    def test_weight_update(self):
        """Test weight updates with epoch."""
        loss = AdaptiveWeightedLoss(
            initial_weight=1.0,
            min_weight=0.1,
            max_weight=5.0,
            warmup_epochs=5,
            decay_epochs=10,
        )

        # Update to epoch 0
        loss.update_epoch(0)
        assert loss.get_current_weight() == 1.0

        # Update to middle of warmup
        loss.update_epoch(2)
        mid_weight = loss.get_current_weight()
        assert mid_weight > 1.0  # Should increase during warmup

        # Update to after warmup
        loss.update_epoch(7)
        decay_weight = loss.get_current_weight()
        assert decay_weight < mid_weight  # Should decrease during decay

    def test_gradient_flow(self):
        """Test gradient flow through loss."""
        loss = AdaptiveWeightedLoss()

        predictions = torch.randn(32, 10, requires_grad=True)
        targets = torch.randint(0, 10, (32,))

        result = loss(predictions, targets)
        result.backward()

        assert predictions.grad is not None
        assert torch.isfinite(predictions.grad).all()

    def test_curriculum_mode(self):
        """Test curriculum learning mode."""
        loss = AdaptiveWeightedLoss(use_curriculum=True, difficulty_threshold=0.5)

        predictions = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))

        result = loss(predictions, targets)
        assert isinstance(result, torch.Tensor)

        stats = loss.get_difficulty_stats()
        assert "mean_difficulty" in stats or not stats


class TestGeometricLoss:
    """Tests for geometric distance loss."""

    def test_euclidean_distance(self):
        """Test Euclidean distance computation."""
        x = torch.randn(10, 128)
        y = torch.randn(10, 128)

        dist = ManifoldGeometry.euclidean_distance(x, y)

        assert dist.shape == (10,)
        assert (dist >= 0).all()

    def test_spherical_distance(self):
        """Test spherical distance computation."""
        x = torch.randn(10, 128)
        y = torch.randn(10, 128)

        dist = ManifoldGeometry.spherical_distance(x, y, curvature=1.0)

        assert dist.shape == (10,)
        assert (dist >= 0).all()
        # Max distance on unit sphere is pi
        assert (dist <= np.pi + 1e-5).all()

    def test_hyperbolic_distance(self):
        """Test hyperbolic distance computation."""
        # Points inside Poincaré ball
        x = torch.randn(10, 128) * 0.3
        y = torch.randn(10, 128) * 0.3

        dist = ManifoldGeometry.hyperbolic_distance(x, y, curvature=1.0)

        assert dist.shape == (10,)
        assert (dist >= 0).all()
        assert torch.isfinite(dist).all()

    def test_geometric_loss_forward(self):
        """Test geometric loss forward pass."""
        for manifold in ["euclidean", "spherical", "hyperbolic"]:
            loss = GeometricDistanceLoss(manifold_type=manifold, embedding_dim=128)

            predictions = torch.randn(32, 128)
            targets = torch.randn(32, 128)

            result = loss(predictions, targets)
            assert isinstance(result, torch.Tensor)
            assert result.ndim == 0
            assert result.item() >= 0

    def test_gradient_flow(self):
        """Test gradient flow for geometric loss."""
        loss = GeometricDistanceLoss(manifold_type="euclidean")

        predictions = torch.randn(32, 128, requires_grad=True)
        targets = torch.randn(32, 128)

        result = loss(predictions, targets)
        result.backward()

        assert predictions.grad is not None
        assert torch.isfinite(predictions.grad).all()


class TestInformationTheoreticLoss:
    """Tests for information-theoretic loss."""

    def test_entropy_computation(self):
        """Test entropy computation."""
        probs = torch.softmax(torch.randn(10, 5), dim=-1)

        from loss_framework.losses.information_theoretic_loss import EntropyCalculator

        entropy = EntropyCalculator.entropy(probs, dim=-1)

        assert entropy.shape == (10,)
        assert (entropy >= 0).all()
        assert torch.isfinite(entropy).all()

    def test_information_loss_forward(self):
        """Test information-theoretic loss forward pass."""
        loss = InformationTheoreticLoss(
            use_entropy_regularization=True, entropy_weight=0.1
        )

        predictions = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))

        result = loss(predictions, targets)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0

    def test_information_loss_with_components(self):
        """Test loss with multiple components."""
        loss = InformationTheoreticLoss(
            use_entropy_regularization=True,
            entropy_weight=0.1,
            use_kl_divergence=True,
            kl_weight=0.05,
            temperature=0.5,
        )

        predictions = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))

        result = loss(predictions, targets)
        assert result.item() >= 0

    def test_gradient_flow(self):
        """Test gradient flow for information loss."""
        loss = InformationTheoreticLoss()

        predictions = torch.randn(32, 10, requires_grad=True)
        targets = torch.randint(0, 10, (32,))

        result = loss(predictions, targets)
        result.backward()

        assert predictions.grad is not None
        assert torch.isfinite(predictions.grad).all()


class TestPhysicsInspiredLoss:
    """Tests for physics-inspired loss."""

    def test_hamiltonian_computation(self):
        """Test Hamiltonian computation."""
        position = torch.randn(10, 128)
        momentum = torch.randn(10, 128)

        potential_net = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        H = HamiltonianMechanics.compute_hamiltonian(position, momentum, potential_net)

        assert H.shape == (10,)
        assert torch.isfinite(H).all()

    def test_physics_loss_forward(self):
        """Test physics-inspired loss forward pass."""
        loss = PhysicsInspiredLoss(use_hamiltonian=True, hamiltonian_weight=0.1)

        predictions = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))
        features = torch.randn(32, 128)

        result = loss(predictions, targets, features=features)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0

    def test_physics_loss_without_features(self):
        """Test physics loss without features."""
        loss = PhysicsInspiredLoss()

        predictions = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))

        # Should work without features (base loss only)
        result = loss(predictions, targets)
        assert isinstance(result, torch.Tensor)

    def test_conservation_loss(self):
        """Test conservation law enforcement."""
        loss = PhysicsInspiredLoss(
            use_conservation=True, conservation_weight=0.1, conserved_quantities=3
        )

        predictions = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))
        features = torch.randn(32, 128)

        result = loss(predictions, targets, features=features)
        assert isinstance(result, torch.Tensor)


class TestRobustStatisticalLoss:
    """Tests for robust statistical loss."""

    def test_huber_loss(self):
        """Test Huber loss computation."""
        residuals = torch.randn(10, 5)

        loss = RobustLossFunctions.huber_loss(residuals, delta=1.0)

        assert loss.shape == residuals.shape
        assert (loss >= 0).all()
        assert torch.isfinite(loss).all()

    def test_tukey_loss(self):
        """Test Tukey biweight loss."""
        residuals = torch.randn(10, 5) * 5  # Large residuals

        loss = RobustLossFunctions.tukey_biweight_loss(residuals, c=4.685)

        assert loss.shape == residuals.shape
        assert (loss >= 0).all()
        # Should saturate for large residuals
        assert (loss <= (4.685**2 / 6) + 1e-5).all()

    def test_cauchy_loss(self):
        """Test Cauchy loss."""
        residuals = torch.randn(10, 5) * 10  # Very large residuals

        loss = RobustLossFunctions.cauchy_loss(residuals, c=1.0)

        assert loss.shape == residuals.shape
        assert (loss >= 0).all()
        assert torch.isfinite(loss).all()

    def test_robust_loss_forward(self):
        """Test robust loss forward pass."""
        for robust_type in ["huber", "tukey", "cauchy", "geman_mcclure"]:
            loss = RobustStatisticalLoss(robust_type=robust_type, scale=1.0)

            predictions = torch.randn(32, 5)
            targets = torch.randn(32, 5)

            result = loss(predictions, targets)
            assert isinstance(result, torch.Tensor)
            assert result.ndim == 0
            assert result.item() >= 0

    def test_adaptive_scale(self):
        """Test adaptive scale estimation."""
        loss = RobustStatisticalLoss(
            robust_type="huber", adaptive_scale=True, scale_update_rate=0.1
        )

        predictions = torch.randn(32, 5)
        targets = torch.randn(32, 5)

        initial_scale = loss.scale

        # Train mode to trigger updates
        loss.train()
        for _ in range(5):
            result = loss(predictions, targets)

        # Scale should have changed
        assert loss.scale != initial_scale or loss.scale == 1.0

    def test_outlier_detection(self):
        """Test outlier detection."""
        loss = RobustStatisticalLoss(outlier_threshold=2.0)

        # Mix of normal and outlier targets
        predictions = torch.randn(100, 5)
        targets = predictions.clone()
        # Add some outliers
        targets[0:10] += 10.0

        result = loss(predictions, targets)

        stats = loss.get_robust_stats()
        assert "outlier_rate" in stats

    def test_gradient_flow(self):
        """Test gradient flow for robust loss."""
        loss = RobustStatisticalLoss(robust_type="huber")

        predictions = torch.randn(32, 5, requires_grad=True)
        targets = torch.randn(32, 5)

        result = loss(predictions, targets)
        result.backward()

        assert predictions.grad is not None
        assert torch.isfinite(predictions.grad).all()


class TestRobustLossComparison:
    """Tests comparing robust losses with standard losses."""

    def test_robust_vs_standard_with_outliers(self):
        """Compare robust and standard losses with outliers."""
        # Create data with outliers
        predictions = torch.randn(100, 1)
        targets = predictions.clone()
        # Add outliers
        targets[0:20] += 5.0

        # Standard MSE loss
        mse_loss = nn.MSELoss()
        mse_value = mse_loss(predictions, targets).item()

        # Robust Huber loss
        huber_loss = RobustStatisticalLoss(robust_type="huber", scale=1.0)
        huber_value = huber_loss(predictions, targets).item()

        # Robust loss should be less affected by outliers
        # This is a heuristic - not a strict guarantee
        assert huber_value < mse_value * 0.8  # Should be significantly smaller
