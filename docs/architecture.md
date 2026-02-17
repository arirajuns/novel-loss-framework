# Architecture Documentation

## System Architecture

### Overview

The Novel Loss Function Framework follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────┐
│         Application Layer               │
│    (Experiments, Training Scripts)      │
├─────────────────────────────────────────┤
│         Loss Function Layer             │
│  (Adaptive, Geometric, Information,     │
│   Physics, Robust Statistical)          │
├─────────────────────────────────────────┤
│         Core Framework Layer            │
│  (BaseLoss, Factory, Registry,          │
│   Composite, Strategy)                  │
├─────────────────────────────────────────┤
│         Configuration Layer             │
│  (Builder Pattern, Validation)          │
├─────────────────────────────────────────┤
│         Utility Layer                   │
│  (Validation, Gradients, Metrics)       │
└─────────────────────────────────────────┘
```

## Design Patterns Implementation

### 1. Template Method Pattern
**Location**: `core/base_loss.py`

The `BaseLoss` class defines the skeleton of loss computation:

```python
class BaseLoss(nn.Module, ABC):
    def forward(self, predictions, targets):
        # Template method
        self._validate_inputs(predictions, targets)
        predictions, targets = self._preprocess_inputs(predictions, targets)
        loss = self._compute_loss(predictions, targets)  # Abstract
        loss = self._apply_reduction(loss)
        loss = self._postprocess_loss(loss)
        self._update_statistics(loss)
        return loss
```

**Hook Methods**:
- `_preprocess_inputs()`: Override for custom preprocessing
- `_postprocess_loss()`: Override for custom postprocessing
- `_compute_loss()`: **Must** implement - core algorithm

### 2. Factory Pattern
**Location**: `core/loss_factory.py`

Centralizes object creation:

```python
class LossFactory:
    @staticmethod
    def create_from_config(config: LossConfig) -> nn.Module:
        # Creates appropriate loss based on config
        
    @staticmethod
    def create_standard(loss_type: str, **kwargs) -> nn.Module:
        # Creates standard PyTorch losses
        
    @staticmethod
    def create_composite(losses: Dict, weights: Dict) -> CompositeLoss:
        # Creates composite loss from multiple losses
```

### 3. Registry Pattern
**Location**: `core/loss_registry.py`

Plugin-style architecture for loss functions:

```python
@register_loss(name="my_loss", category="custom")
class MyLoss(BaseLoss):
    pass

# Later retrieval
loss_class = LossRegistry.get("my_loss")
loss = LossRegistry.create("my_loss", **kwargs)
```

**Benefits**:
- Dynamic discovery of loss functions
- No hard-coded mappings
- Easy extension with new losses

### 4. Strategy Pattern
**Location**: `losses/adaptive_weighted_loss.py`, `losses/geometric_loss.py`

Interchangeable algorithms:

```python
class WeightScheduleStrategy:
    @staticmethod
    def linear_schedule(epoch, ...): ...
    
    @staticmethod
    def exponential_schedule(epoch, ...): ...
    
    @staticmethod
    def cosine_schedule(epoch, ...): ...

# Usage
schedule_fn = self._get_schedule_function()
weight = schedule_fn(epoch, ...)
```

### 5. Composite Pattern
**Location**: `core/composite_loss.py`

Combines multiple losses:

```python
composite = CompositeLoss({
    'mse': nn.MSELoss(),
    'l1': nn.L1Loss()
}, weights={'mse': 0.7, 'l1': 0.3})

total_loss = composite(predictions, targets)
```

### 6. Builder Pattern
**Location**: `config/`

Step-by-step construction of complex configurations:

```python
@dataclass
class ExperimentConfig(BaseConfig):
    loss_config: LossConfig
    model_config: ModelConfig
    training_config: TrainingConfig
    # ... composed configurations
```

## SOLID Principles Compliance

### Single Responsibility Principle (SRP)
✅ **BaseLoss**: Only defines loss computation structure
✅ **LossFactory**: Only creates loss instances
✅ **LossRegistry**: Only manages registration
✅ **InputValidator**: Only validates inputs

### Open/Closed Principle (OCP)
✅ **BaseLoss**: Open for extension via hooks, closed for modification
✅ **LossRegistry**: New losses added via decorator without changing registry
✅ **WeightScheduleStrategy**: New schedules added without changing existing

### Liskov Substitution Principle (LSP)
✅ All loss functions can substitute BaseLoss
✅ All configs can substitute BaseConfig
✅ Factory returns can be used interchangeably

### Interface Segregation Principle (ISP)
✅ Small, focused interfaces
✅ `BaseLoss` has minimal required interface
✅ Config classes inherit from focused base

### Dependency Inversion Principle (DIP)
✅ High-level modules depend on abstractions
✅ `BaseLoss` is abstract base class
✅ `BaseConfig` is abstract base class
✅ Loss functions depend on interfaces, not concrete classes

## Configuration System

### Hierarchy

```
BaseConfig (Abstract)
├── LossConfig
│   ├── AdaptiveLossConfig
│   ├── GeometricLossConfig
│   ├── InformationTheoreticLossConfig
│   ├── PhysicsInspiredLossConfig
│   └── RobustStatisticalLossConfig
├── ModelConfig
├── TrainingConfig
├── DataConfig
├── LoggingConfig
└── EvaluationConfig

ExperimentConfig (Aggregates all above)
```

### Features
- Validation on initialization (`__post_init__`)
- Serialization (JSON, YAML)
- Freezing to prevent modification
- Merging configurations
- Copying

## Testing Architecture

### Test Organization
```
tests/
├── conftest.py          # Fixtures and configuration
├── test_config.py       # Configuration system tests
├── test_core.py         # Core framework tests
├── test_novel_losses.py # Novel loss function tests
└── test_integration.py  # End-to-end tests
```

### Testing Strategy
1. **Unit Tests**: Individual components
2. **Integration Tests**: Component interactions
3. **Property Tests**: Mathematical properties
4. **Regression Tests**: Known good outputs

### Test Coverage Goals
- Core framework: >90%
- Novel losses: >85%
- Utilities: >80%
- Integration: All major workflows

## Extension Guidelines

### Adding a New Loss Function

1. **Create file**: `losses/my_loss.py`

2. **Implement class**:
```python
from ..core.base_loss import BaseLoss
from ..core.loss_registry import register_loss

@register_loss(name="my_loss", category="custom")
class MyLoss(BaseLoss):
    def _compute_loss(self, predictions, targets, **kwargs):
        # Implementation
        return loss
```

3. **Create config** (optional):
```python
@dataclass
class MyLossConfig(LossConfig):
    loss_type: str = "my_loss"
    my_param: float = 1.0
    
    def validate(self):
        super().validate()
        if self.my_param <= 0:
            raise ValueError("my_param must be positive")
```

4. **Add tests**: `tests/test_my_loss.py`

5. **Document**: Update README and architecture docs

### Best Practices
- ✅ Use Template Method hooks
- ✅ Add input validation
- ✅ Track statistics
- ✅ Write comprehensive tests
- ✅ Document mathematical properties
- ✅ Follow existing code style

## Performance Considerations

### Memory Optimization
- Use `torch.no_grad()` for statistics tracking
- Clear history periodically
- Use lazy initialization where possible

### Computational Optimization
- Vectorized operations
- GPU acceleration support
- Batch processing

### Gradient Flow
- All losses tested for gradient flow
- NaN/Inf detection
- Gradient clipping support