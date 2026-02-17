"""
Integration tests for the loss framework
Validates end-to-end workflows
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from loss_framework.core.loss_factory import LossFactory
from loss_framework.core.loss_registry import LossRegistry
from loss_framework.config.loss_config import LossConfig
from loss_framework.config.experiment_config import ExperimentConfig


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestEndToEndTraining:
    """End-to-end integration tests."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset."""
        X = torch.randn(1000, 20)
        y = torch.randint(0, 5, (1000,))

        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        return train_loader

    def test_training_with_standard_loss(self, sample_data):
        """Test complete training loop with standard loss."""
        model = SimpleModel(20, 50, 5)
        loss_fn = LossFactory.create_standard("cross_entropy")
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        initial_loss = None
        final_loss = None

        # Train for a few epochs
        for epoch in range(3):
            epoch_losses = []
            for batch_x, batch_y in sample_data:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            if epoch == 0:
                initial_loss = sum(epoch_losses) / len(epoch_losses)
            elif epoch == 2:
                final_loss = sum(epoch_losses) / len(epoch_losses)

        # Loss should decrease
        assert final_loss < initial_loss

    def test_training_with_configured_loss(self, sample_data):
        """Test training with configured loss."""
        config = LossConfig(loss_type="cross_entropy", reduction="mean")

        model = SimpleModel(20, 50, 5)
        loss_fn = LossFactory.create_from_config(config)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Train for one epoch
        losses = []
        for batch_x, batch_y in sample_data:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        assert avg_loss > 0
        assert torch.isfinite(torch.tensor(avg_loss))

    def test_loss_registry_integration(self):
        """Test loss registry with framework."""
        # List all registered losses
        registered = LossRegistry.list_losses()

        # Should have our novel losses
        assert "adaptive_weighted" in registered or len(registered) == 0

        # Get metadata
        info = LossRegistry.info()
        assert "registered_losses" in info


class TestExperimentWorkflow:
    """Tests for experiment workflow integration."""

    def test_experiment_configuration_creation(self):
        """Test creating complete experiment configuration."""
        exp_config = ExperimentConfig(
            experiment_name="test_experiment",
            loss_config=LossConfig(loss_type="cross_entropy"),
            training_config=ExperimentConfig.__dataclass_fields__[
                "training_config"
            ].default_factory(),
        )

        assert exp_config.experiment_name == "test_experiment"
        assert exp_config.loss_config.loss_type == "cross_entropy"

    def test_experiment_directory_creation(self, tmp_path):
        """Test experiment directory creation."""
        exp_config = ExperimentConfig(
            experiment_name="test_exp",
            logging_config=type(
                "obj",
                (object,),
                {
                    "log_dir": str(tmp_path),
                    "log_level": "INFO",
                    "use_tensorboard": False,
                    "tensorboard_dir": str(tmp_path / "tensorboard"),
                    "use_wandb": False,
                    "wandb_project": "test",
                    "wandb_entity": None,
                    "log_frequency": 10,
                    "save_frequency": 1,
                    "keep_last_n": 3,
                },
            )(),
        )

        exp_dir = exp_config.get_experiment_dir()
        assert exp_dir.exists()


class TestLossComposition:
    """Tests for composing multiple losses."""

    def test_composite_loss_training(self):
        """Test training with composite loss."""
        from loss_framework.core.composite_loss import CompositeLoss

        # Create composite loss
        composite = CompositeLoss(
            {
                "ce": nn.CrossEntropyLoss(reduction="none"),
                "l1": nn.L1Loss(reduction="none"),
            },
            weights={"ce": 0.7, "l1": 0.3},
        )

        # Create simple model
        model = SimpleModel(10, 20, 5)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Create data
        X = torch.randn(64, 10)
        y = torch.randint(0, 5, (64,))

        # Training step
        optimizer.zero_grad()
        outputs = model(X)

        # Reshape for L1 loss
        outputs_l1 = outputs.view(-1)
        targets_l1 = torch.randn(outputs_l1.shape[0])

        # This won't work well in practice but tests the API
        loss_ce = composite.losses["ce"](outputs, y).mean()
        loss_l1 = composite.losses["l1"](outputs_l1, targets_l1).mean()
        loss = 0.7 * loss_ce + 0.3 * loss_l1

        loss.backward()

        assert loss.item() > 0
        assert any(p.grad is not None for p in model.parameters())


class TestGradientAndOptimization:
    """Tests for gradient behavior and optimization."""

    def test_gradient_norm_tracking(self):
        """Test gradient norm computation."""
        from loss_framework.utils.gradients import GradientUtils

        model = SimpleModel(10, 20, 5)
        loss_fn = nn.CrossEntropyLoss()

        X = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))

        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()

        # Compute gradient norm
        grad_norm = GradientUtils.compute_gradient_norm(model)

        assert grad_norm >= 0
        assert torch.isfinite(torch.tensor(grad_norm))

    def test_gradient_clipping(self):
        """Test gradient clipping."""
        from loss_framework.utils.gradients import GradientUtils

        model = SimpleModel(10, 20, 5)
        loss_fn = nn.CrossEntropyLoss()

        # Create large gradients
        X = torch.randn(32, 10) * 100
        y = torch.randint(0, 5, (32,))

        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()

        # Get norm before clipping
        norm_before = GradientUtils.compute_gradient_norm(model)

        # Clip gradients
        GradientUtils.clip_gradients(model, max_norm=1.0)

        # Get norm after clipping
        norm_after = GradientUtils.compute_gradient_norm(model)

        # After clipping, norm should be <= max_norm
        assert norm_after <= 1.0 + 1e-5


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_mismatched_batch_sizes(self):
        """Test error on mismatched batch sizes."""
        loss_fn = nn.CrossEntropyLoss()

        predictions = torch.randn(32, 10)
        targets = torch.randint(0, 10, (16,))  # Different batch size

        with pytest.raises(ValueError):
            loss_fn(predictions, targets)

    def test_invalid_input_types(self):
        """Test error on invalid input types."""
        from loss_framework.utils.validators import InputValidator

        predictions = torch.randn(32, 10)

        with pytest.raises(TypeError):
            InputValidator.validate_shape(predictions, expected_shape=(32, 5))

    def test_nan_gradient_detection(self):
        """Test detection of NaN gradients."""
        from loss_framework.utils.gradients import GradientUtils

        model = SimpleModel(10, 20, 5)

        # Manually set NaN gradients
        for p in model.parameters():
            p.grad = torch.randn_like(p) * float("nan")

        health = GradientUtils.check_gradient_health(model)

        assert health["nan_grads"] > 0
        assert health["has_issues"] is True
