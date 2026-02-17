"""
Tests for configuration system
Validates Builder pattern implementation
"""

import pytest
import torch
from loss_framework.config.base_config import BaseConfig
from loss_framework.config.loss_config import (
    LossConfig,
    AdaptiveLossConfig,
    GeometricLossConfig,
    InformationTheoreticLossConfig,
    PhysicsInspiredLossConfig,
    RobustStatisticalLossConfig,
)
from loss_framework.config.experiment_config import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    LoggingConfig,
    EvaluationConfig,
)


class TestBaseConfig:
    """Tests for base configuration."""

    def test_config_validation(self):
        """Test configuration validation."""
        # This should fail as BaseConfig is abstract
        with pytest.raises(TypeError):
            BaseConfig()

    def test_config_to_dict(self):
        """Test conversion to dictionary."""
        config = LossConfig(loss_type="mse", reduction="sum")
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["loss_type"] == "mse"
        assert config_dict["reduction"] == "sum"

    def test_config_freeze(self):
        """Test configuration freezing."""
        config = LossConfig()
        config.freeze()

        with pytest.raises(AttributeError):
            config.loss_type = "new_type"

    def test_config_copy(self):
        """Test configuration copying."""
        config = LossConfig(loss_type="cross_entropy")
        config_copy = config.copy()

        assert config_copy.loss_type == config.loss_type
        assert config_copy is not config


class TestLossConfig:
    """Tests for loss configurations."""

    def test_default_loss_config(self):
        """Test default loss configuration."""
        config = LossConfig()

        assert config.loss_type == "cross_entropy"
        assert config.reduction == "mean"
        assert config.device == "auto"

    def test_invalid_reduction(self):
        """Test validation of reduction parameter."""
        with pytest.raises(ValueError):
            LossConfig(reduction="invalid")

    def test_invalid_loss_scale(self):
        """Test validation of loss scale."""
        with pytest.raises(ValueError):
            LossConfig(loss_scale=0)

        with pytest.raises(ValueError):
            LossConfig(loss_scale=-1)

    def test_adaptive_loss_config(self):
        """Test adaptive loss configuration."""
        config = AdaptiveLossConfig(
            schedule_type="cosine", warmup_epochs=10, decay_epochs=90
        )

        assert config.schedule_type == "cosine"
        assert config.warmup_epochs == 10
        assert config.decay_epochs == 90

    def test_invalid_schedule_type(self):
        """Test validation of schedule type."""
        with pytest.raises(ValueError):
            AdaptiveLossConfig(schedule_type="invalid")

    def test_geometric_loss_config(self):
        """Test geometric loss configuration."""
        config = GeometricLossConfig(manifold_type="spherical", curvature=2.0)

        assert config.manifold_type == "spherical"
        assert config.curvature == 2.0

    def test_invalid_manifold_type(self):
        """Test validation of manifold type."""
        with pytest.raises(ValueError):
            GeometricLossConfig(manifold_type="invalid")

    def test_information_theoretic_config(self):
        """Test information-theoretic loss configuration."""
        config = InformationTheoreticLossConfig(entropy_weight=0.2, temperature=0.5)

        assert config.entropy_weight == 0.2
        assert config.temperature == 0.5

    def test_physics_inspired_config(self):
        """Test physics-inspired loss configuration."""
        config = PhysicsInspiredLossConfig(use_hamiltonian=True, hamiltonian_weight=0.1)

        assert config.use_hamiltonian is True
        assert config.hamiltonian_weight == 0.1

    def test_robust_statistical_config(self):
        """Test robust statistical loss configuration."""
        config = RobustStatisticalLossConfig(robust_type="tukey", adaptive_scale=True)

        assert config.robust_type == "tukey"
        assert config.adaptive_scale is True

    def test_invalid_robust_type(self):
        """Test validation of robust type."""
        with pytest.raises(ValueError):
            RobustStatisticalLossConfig(robust_type="invalid")


class TestExperimentConfig:
    """Tests for experiment configuration."""

    def test_default_experiment_config(self):
        """Test default experiment configuration."""
        config = ExperimentConfig()

        assert config.experiment_name is not None
        assert config.experiment_id is not None
        assert isinstance(config.loss_config, LossConfig)
        assert isinstance(config.model_config, ModelConfig)
        assert isinstance(config.training_config, TrainingConfig)

    def test_experiment_config_validation(self):
        """Test experiment configuration validation."""
        config = ExperimentConfig(training_config=TrainingConfig(num_epochs=50))

        # Should validate without error
        config.validate()
        assert config.training_config.num_epochs == 50

    def test_invalid_training_config(self):
        """Test validation of training configuration."""
        with pytest.raises(ValueError):
            ExperimentConfig(training_config=TrainingConfig(num_epochs=0))

    def test_experiment_save_load(self, tmp_path):
        """Test saving and loading experiment configuration."""
        config = ExperimentConfig(
            experiment_name="test_exp", loss_config=LossConfig(loss_type="mse")
        )

        # Save
        filepath = tmp_path / "config.yaml"
        config.save(str(filepath))

        # Load
        loaded_config = ExperimentConfig.load(str(filepath))

        assert loaded_config.experiment_name == "test_exp"
        assert loaded_config.loss_config.loss_type == "mse"


class TestDataConfig:
    """Tests for data configuration."""

    def test_valid_data_split(self):
        """Test valid data split configuration."""
        config = DataConfig(train_split=0.7, val_split=0.15, test_split=0.15)

        assert config.train_split == 0.7
        assert config.val_split == 0.15
        assert config.test_split == 0.15

    def test_invalid_data_split(self):
        """Test invalid data split configuration."""
        with pytest.raises(ValueError):
            DataConfig(
                train_split=0.5,
                val_split=0.3,
                test_split=0.3,  # Sums to 1.1
            )
