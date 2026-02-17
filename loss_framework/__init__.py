# Novel Loss Function Framework
# Comprehensive framework for implementing and testing novel loss functions in PyTorch
# Using SOLID principles and design patterns

__version__ = "1.0.0"
__author__ = "arirajuns"

from .core.base_loss import BaseLoss
from .core.loss_factory import LossFactory
from .core.loss_registry import LossRegistry
from .config.loss_config import LossConfig
from .config.experiment_config import ExperimentConfig

__all__ = ["BaseLoss", "LossFactory", "LossRegistry", "LossConfig", "ExperimentConfig"]
