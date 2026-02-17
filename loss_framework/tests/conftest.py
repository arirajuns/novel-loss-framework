"""
Test suite for loss functions
Comprehensive testing using pytest
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


# Test fixtures
@pytest.fixture
def sample_batch():
    """Create sample prediction and target tensors."""
    batch_size = 32
    num_classes = 10

    # Random logits and labels
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    return predictions, targets


@pytest.fixture
def sample_regression_batch():
    """Create sample regression data."""
    batch_size = 32

    predictions = torch.randn(batch_size, 1)
    targets = torch.randn(batch_size, 1)

    return predictions, targets


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
