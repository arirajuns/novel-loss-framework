"""
Tests for core loss framework
Validates Factory, Registry, and Template Method patterns
"""

import pytest
import torch
import torch.nn as nn
from loss_framework.core.base_loss import BaseLoss
from loss_framework.core.loss_factory import LossFactory
from loss_framework.core.loss_registry import LossRegistry, register_loss
from loss_framework.core.composite_loss import CompositeLoss
from loss_framework.config.loss_config import LossConfig


class TestBaseLoss:
    """Tests for abstract base loss class."""

    def test_base_loss_is_abstract(self):
        """Test that BaseLoss cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLoss()

    def test_concrete_loss_instantiation(self):
        """Test creating a concrete loss implementation."""

        class ConcreteLoss(BaseLoss):
            def _compute_loss(self, predictions, targets, **kwargs):
                return torch.mean((predictions - targets) ** 2)

        loss = ConcreteLoss()
        assert isinstance(loss, nn.Module)

    def test_loss_forward(self):
        """Test loss forward pass."""

        class ConcreteLoss(BaseLoss):
            def _compute_loss(self, predictions, targets, **kwargs):
                return torch.mean((predictions - targets) ** 2)

        loss = ConcreteLoss()
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)

        result = loss(predictions, targets)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0  # Scalar


class TestLossRegistry:
    """Tests for loss registry pattern."""

    def test_register_loss(self):
        """Test registering a loss function."""

        @register_loss(name="test_loss")
        class TestLoss(BaseLoss):
            def _compute_loss(self, predictions, targets, **kwargs):
                return torch.mean((predictions - targets) ** 2)

        # Check registration
        assert "test_loss" in LossRegistry.list_losses()

    def test_get_registered_loss(self):
        """Test getting a registered loss class."""

        @register_loss(name="another_test_loss")
        class AnotherTestLoss(BaseLoss):
            def _compute_loss(self, predictions, targets, **kwargs):
                return torch.mean((predictions - targets) ** 2)

        loss_class = LossRegistry.get("another_test_loss")
        assert loss_class is not None
        assert loss_class.__name__ == "AnotherTestLoss"

    def test_create_registered_loss(self):
        """Test creating instance of registered loss."""

        @register_loss(name="create_test_loss")
        class CreateTestLoss(BaseLoss):
            def _compute_loss(self, predictions, targets, **kwargs):
                return torch.mean((predictions - targets) ** 2)

        loss = LossRegistry.create("create_test_loss")
        assert isinstance(loss, BaseLoss)

    def test_unknown_loss_error(self):
        """Test error for unknown loss."""
        with pytest.raises(ValueError):
            LossRegistry.create("nonexistent_loss")


class TestLossFactory:
    """Tests for loss factory pattern."""

    def test_create_standard_loss(self):
        """Test creating standard PyTorch losses."""
        loss = LossFactory.create_standard("mse")
        assert isinstance(loss, nn.MSELoss)

        loss = LossFactory.create_standard("cross_entropy")
        assert isinstance(loss, nn.CrossEntropyLoss)

        loss = LossFactory.create_standard("l1")
        assert isinstance(loss, nn.L1Loss)

    def test_create_from_config(self):
        """Test creating loss from configuration."""
        config = LossConfig(loss_type="mse", reduction="sum")
        loss = LossFactory.create_from_config(config)

        assert isinstance(loss, nn.MSELoss)
        assert loss.reduction == "sum"

    def test_create_from_dict(self):
        """Test creating loss from dictionary."""
        config_dict = {"loss_type": "l1", "reduction": "mean"}
        loss = LossFactory.create_from_dict(config_dict)

        assert isinstance(loss, nn.L1Loss)

    def test_invalid_loss_type(self):
        """Test error for invalid loss type."""
        with pytest.raises(ValueError):
            LossFactory.create_standard("invalid_loss")


class TestCompositeLoss:
    """Tests for composite loss pattern."""

    def test_composite_loss_creation(self):
        """Test creating composite loss."""
        losses = {
            "mse": nn.MSELoss(reduction="none"),
            "l1": nn.L1Loss(reduction="none"),
        }
        weights = {"mse": 0.7, "l1": 0.3}

        composite = CompositeLoss(losses, weights)
        assert len(composite.losses) == 2

    def test_composite_loss_forward(self):
        """Test composite loss forward pass."""
        losses = {
            "mse": nn.MSELoss(reduction="none"),
            "l1": nn.L1Loss(reduction="none"),
        }

        composite = CompositeLoss(losses)

        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)

        loss = composite(predictions, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_composite_loss_weights(self):
        """Test composite loss weight management."""
        losses = {"mse": nn.MSELoss(reduction="none")}

        composite = CompositeLoss(losses, weights={"mse": 1.0})
        assert composite.get_weights()["mse"] == 1.0

        # Update weight
        composite.set_weight("mse", 2.0)
        assert composite.get_weights()["mse"] == 2.0

    def test_composite_add_remove_loss(self):
        """Test adding and removing losses."""
        composite = CompositeLoss({"mse": nn.MSELoss(reduction="none")})

        # Add loss
        composite.add_loss("l1", nn.L1Loss(reduction="none"), weight=0.5)
        assert "l1" in composite.loss_names

        # Remove loss
        composite.remove_loss("l1")
        assert "l1" not in composite.loss_names


class TestTemplateMethodPattern:
    """Tests validating Template Method pattern implementation."""

    def test_preprocessing_hook(self):
        """Test preprocessing hook is called."""
        preprocessing_called = False

        class HookedLoss(BaseLoss):
            def _preprocess_inputs(self, predictions, targets):
                nonlocal preprocessing_called
                preprocessing_called = True
                return predictions * 2, targets * 2

            def _compute_loss(self, predictions, targets, **kwargs):
                return torch.mean(predictions - targets)

        loss = HookedLoss()
        predictions = torch.ones(5)
        targets = torch.zeros(5)

        loss(predictions, targets)
        assert preprocessing_called is True

    def test_postprocessing_hook(self):
        """Test postprocessing hook is called."""
        postprocessing_called = False

        class HookedLoss(BaseLoss):
            def _postprocess_loss(self, loss):
                nonlocal postprocessing_called
                postprocessing_called = True
                return loss * 2

            def _compute_loss(self, predictions, targets, **kwargs):
                return torch.mean((predictions - targets) ** 2)

        loss = HookedLoss()
        predictions = torch.randn(5)
        targets = torch.randn(5)

        result = loss(predictions, targets)
        assert postprocessing_called is True

    def test_reduction_methods(self):
        """Test different reduction methods."""

        class TestLoss(BaseLoss):
            def _compute_loss(self, predictions, targets, **kwargs):
                return (predictions - targets) ** 2

        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)

        # Test mean reduction
        loss_mean = TestLoss(reduction="mean")
        result_mean = loss_mean(predictions, targets)
        assert result_mean.ndim == 0

        # Test sum reduction
        loss_sum = TestLoss(reduction="sum")
        result_sum = loss_sum(predictions, targets)
        assert result_sum.ndim == 0

        # Test none reduction
        loss_none = TestLoss(reduction="none")
        result_none = loss_none(predictions, targets)
        assert result_none.shape == predictions.shape


class TestLossStatistics:
    """Tests for loss statistics tracking."""

    def test_statistics_tracking(self):
        """Test that statistics are tracked."""

        class TestLoss(BaseLoss):
            def _compute_loss(self, predictions, targets, **kwargs):
                return torch.mean((predictions - targets) ** 2)

        loss = TestLoss()

        # Make multiple forward passes
        for _ in range(5):
            predictions = torch.randn(10, 5)
            targets = torch.randn(10, 5)
            loss(predictions, targets)

        stats = loss.get_statistics()
        assert stats["call_count"] == 5
        assert "avg_loss" in stats

    def test_statistics_reset(self):
        """Test statistics reset."""

        class TestLoss(BaseLoss):
            def _compute_loss(self, predictions, targets, **kwargs):
                return torch.mean((predictions - targets) ** 2)

        loss = TestLoss()

        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)
        loss(predictions, targets)

        loss.reset_statistics()
        stats = loss.get_statistics()
        assert stats["call_count"] == 0
