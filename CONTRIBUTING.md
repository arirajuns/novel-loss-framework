# Contributing to Novel Loss Function Framework

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code:
- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Respect different viewpoints and experiences

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Use a clear descriptive title**
- **Describe the exact steps to reproduce**
- **Provide specific examples** (code snippets, stack traces)
- **Describe the behavior you observed and expected**
- **Include your environment details** (OS, Python version, PyTorch version)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Include:

- **Use a clear descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples** to demonstrate the enhancement
- **Explain why this enhancement would be useful**

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest loss_framework/tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/novel-loss-framework.git
cd novel-loss-framework

# Create conda environment
conda env create -f environment.yml
conda activate novel-loss-framework

# Install in development mode
pip install -e .

# Run tests
pytest loss_framework/tests/ -v
```

## Style Guidelines

### Python Code Style

We follow PEP 8 with these specifics:
- **Line length**: Maximum 100 characters
- **Imports**: Grouped as standard library, third-party, local
- **Type hints**: Required for all function signatures
- **Docstrings**: Google style docstrings

```python
def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weight: float = 1.0
) -> torch.Tensor:
    """Compute weighted loss.
    
    Args:
        predictions: Model predictions with shape (batch_size, num_classes)
        targets: Ground truth labels with shape (batch_size,)
        weight: Loss weight multiplier
        
    Returns:
        Scalar loss value
        
    Raises:
        ValueError: If predictions and targets have mismatched batch sizes
    """
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError("Batch size mismatch")
    # ... implementation
```

### Git Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and PRs where appropriate

### Testing

All contributions must include tests:

```python
def test_new_feature():
    """Test description."""
    # Arrange
    input_data = torch.randn(10, 5)
    expected_output = torch.tensor([...])
    
    # Act
    result = new_feature(input_data)
    
    # Assert
    assert torch.allclose(result, expected_output)
```

Run tests before submitting:
```bash
pytest loss_framework/tests/ -v --cov=loss_framework
```

### Documentation

- Update README.md if adding new features
- Add docstrings to all public APIs
- Update relevant .md files in docs/ directory
- Include usage examples for new loss functions

## Adding New Loss Functions

To add a new loss function:

1. **Create implementation** in `loss_framework/losses/your_loss.py`
2. **Inherit from BaseLoss** following existing patterns
3. **Add tests** in `loss_framework/tests/test_novel_losses.py`
4. **Register in factory** in `loss_framework/core/loss_factory.py`
5. **Add configuration** in `loss_framework/config/loss_config.py`
6. **Update README** with usage example
7. **Add benchmark** comparing to baseline

Example structure:

```python
from loss_framework.core.base_loss import BaseLoss

class YourNovelLoss(BaseLoss):
    """Your loss function description.
    
    Mathematical formulation:
        L = ...
    
    Reference:
        Author et al. (Year). Paper Title. Conference/Journal.
    """
    
    def __init__(self, param1: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.param1 = param1
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        # Implementation
        pass
```

## Review Process

Pull requests require:
- At least one review from a maintainer
- All CI checks must pass
- Code coverage must not decrease
- Documentation must be updated

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

- Open an issue for questions
- Check existing documentation in docs/
- Review closed issues for similar questions

Thank you for contributing!
