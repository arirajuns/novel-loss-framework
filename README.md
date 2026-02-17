# Novel Loss Function Framework

[![CI](https://github.com/yourusername/novel-loss-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/novel-loss-framework/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

## Executive Summary

A comprehensive framework for implementing and testing novel loss functions in PyTorch. Built with solid software engineering principles including SOLID, design patterns, and quality assurance methodologies (DMADV, DMAIC, PDCA).

**⚠️ Important**: This is a research and educational framework. See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for detailed performance benchmarks and limitations.

## Project Structure

```
loss_framework/
├── config/              # Configuration classes with Builder pattern
├── core/                # Core framework (Factory, Registry, Composite)
├── losses/              # Novel loss function implementations
├── utils/               # Utilities (validation, gradients, metrics)
├── tests/               # Comprehensive test suite
├── experiments/         # Experiment logs and results
└── docs/               # Documentation

Novel Loss Functions Implemented:
1. Adaptive Weighted Loss - Curriculum learning with dynamic weight adjustment
2. Geometric Distance Loss - Riemannian geometry on manifolds
3. Information-Theoretic Loss - Entropy regularization and MI maximization
4. Physics-Inspired Loss - Hamiltonian mechanics and conservation laws
5. Robust Statistical Loss - M-estimators and adaptive trimming
```

## Design Patterns Used

1. **Factory Pattern** - `LossFactory` for creating loss instances
2. **Registry Pattern** - `LossRegistry` for loss management
3. **Template Method Pattern** - `BaseLoss` with hooks
4. **Strategy Pattern** - Weight scheduling and manifold geometry
5. **Composite Pattern** - `CompositeLoss` for combining losses
6. **Builder Pattern** - Configuration system
7. **Singleton Pattern** - Registry instances

## Quality Methodologies

### DMADV (Define, Measure, Analyze, Design, Verify)
- **Define**: Project scope and requirements
- **Measure**: Performance metrics, gradient flow analysis
- **Analyze**: Mathematical properties, convergence
- **Design**: Software architecture
- **Verify**: Unit and integration tests

### DMAIC for Each Loss Function
- **Define**: Mathematical formulation
- **Measure**: Gradient behavior, loss landscapes
- **Analyze**: Compare with baselines
- **Improve**: Hyperparameter tuning
- **Control**: Standardized testing

### PDCA (Plan-Do-Check-Act)
- **Plan**: Experiment design
- **Do**: Execute training runs
- **Check**: Analyze results
- **Act**: Iterate on design

## Installation

### Option 1: Using Python venv (Recommended)

```bash
# Clone repository
git clone https://github.com/arirajuns/novel-loss-framework.git
cd novel-loss-framework

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Option 2: Using Conda

```bash
# Clone repository
git clone https://github.com/arirajuns/novel-loss-framework.git
cd novel-loss-framework

# Create conda environment
conda env create -f environment.yml
conda activate novel-loss-framework

# Install package in development mode
pip install -e .
```

### Option 3: Direct pip install from GitHub

```bash
# Install directly from GitHub (no cloning required)
pip install git+https://github.com/arirajuns/novel-loss-framework.git
```

### Verify Installation

```bash
# Run tests
pytest loss_framework/tests/ -v

# Check 87% of tests should pass (68/78)
```

## Quick Start

```python
from loss_framework import LossFactory, LossConfig
from loss_framework.losses import AdaptiveWeightedLoss

# Create loss from config
config = LossConfig(loss_type='adaptive_weighted')
loss = LossFactory.create_from_config(config)

# Or directly
loss = AdaptiveWeightedLoss(
    base_loss='cross_entropy',
    schedule_type='cosine',
    warmup_epochs=10
)

# Use in training
predictions = model(inputs)
loss_value = loss(predictions, targets)
loss_value.backward()
```

## Novel Loss Functions

### 1. Adaptive Weighted Loss
- Dynamic weight adjustment based on training progress
- Multiple scheduling strategies (linear, exponential, cosine)
- Curriculum learning support
- Usage: `AdaptiveWeightedLoss(schedule_type='cosine')`

### 2. Geometric Distance Loss
- Euclidean, spherical, and hyperbolic manifolds
- Geodesic and chordal distances
- Hierarchical data representation
- Usage: `GeometricDistanceLoss(manifold_type='spherical')`

### 3. Information-Theoretic Loss
- Entropy regularization
- Mutual information maximization
- KL divergence constraints
- Usage: `InformationTheoreticLoss(entropy_weight=0.1)`

### 4. Physics-Inspired Loss
- Hamiltonian dynamics regularization
- Conservation law enforcement
- Lagrangian mechanics
- Usage: `PhysicsInspiredLoss(use_hamiltonian=True)`

### 5. Robust Statistical Loss
- Huber, Tukey, Cauchy loss functions
- Adaptive scale estimation
- Outlier detection and trimming
- Usage: `RobustStatisticalLoss(robust_type='tukey')`

## Testing

```bash
# Run all tests
pytest loss_framework/tests/ -v

# Run specific test category
pytest loss_framework/tests/test_novel_losses.py -v

# With coverage
pytest --cov=loss_framework tests/
```

## Experiment Logging

Experiments are automatically logged with:
- Configuration snapshots
- Loss statistics
- Gradient flow analysis
- Performance metrics
- Git commit tracking

Logs stored in: `experiments/logs/`

## Known Limitations

### Performance
- **4-9x computational overhead** compared to PyTorch built-in losses
- **1.5-2.3x higher memory usage**
- Tested only on small-medium datasets (MNIST, IMDB, synthetic)

### Implementation
- **Test Coverage**: 91% (71/78 tests passing)
- Some device handling issues in CompositeLoss
- Hyperbolic distance can be numerically unstable near boundaries
- PhysicsInspired loss occasionally shows gradient spikes

### Validation
- Only tested on classification tasks
- Single GPU testing only
- No production-scale validation (>100K samples)

**See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for detailed analysis.**

### Known Issues & Troubleshooting

For detailed information on:
- All 7 test failures and their root causes
- Workarounds for common issues
- Solutions and planned fixes
- Recommended usage patterns

**See [KNOWN_ISSUES.md](KNOWN_ISSUES.md)**

**Quick Fixes for Common Issues:**
1. **CompositeLoss fails with "Tensor cannot be converted to Scalar"** → Use `reduction='mean'` in sub-losses
2. **Device mismatch errors** → Ensure all tensors are on same device with `.to(device)`
3. **Config serialization fails** → Avoid YAML save/load, use direct Python objects
4. **Weight schedule not as expected** → Explicitly set `schedule_type='linear'`

## When to Use

### ✅ Recommended
- Research and experimentation
- Educational purposes
- Datasets with label noise (>10%)
- Highly imbalanced classes
- When 2-6% accuracy improvement justifies 4-9x slowdown

### ❌ Not Recommended
- Production systems without validation
- Real-time inference (speed critical)
- Resource-constrained environments
- Clean, balanced datasets (use PyTorch built-ins instead)

## Contributing

1. Follow SOLID principles
2. Add comprehensive tests
3. Document all public APIs
4. Use type hints
5. Follow existing code style

## License

MIT License - See LICENSE file for details

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{novel_loss_framework,
  title={Novel Loss Function Framework},
  author={AI Development Team},
  year={2026},
  url={https://github.com/yourusername/novel-loss-framework}
}
```
