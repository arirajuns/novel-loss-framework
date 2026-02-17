# Known Issues and Troubleshooting Guide

This document lists all known issues with the Novel Loss Function Framework, their root causes, and workarounds/solutions.

## ⚠️ Current Test Status

**Overall**: 71/78 tests passing (91% success rate)

**Failed Tests**: 7 out of 78 total tests

---

## 🔴 Critical Issues

### Issue 1: Config Serialization (2 test failures)

**Affected Tests**:
- `test_experiment_save_load`
- `test_experiment_directory_creation`

**Error Message**:
```
AttributeError: 'dict' object has no attribute 'validate'
```

**Root Cause**: 
When loading YAML configurations, nested config objects (like `LossConfig`, `LoggingConfig`) are loaded as plain dictionaries instead of being converted back to their respective class instances. The `from_yaml()` method doesn't properly reconstruct nested dataclass objects.

**Impact**: 
- Cannot save/load experiment configurations from YAML files
- Breaks experiment reproducibility features

**Workaround**:
```python
# Instead of using YAML serialization:
config = ExperimentConfig.load("config.yaml")  # ❌ Doesn't work

# Use direct Python objects:
config = ExperimentConfig(
    experiment_name="my_exp",
    loss_config=LossConfig(loss_type="cross_entropy")
)  # ✅ Works fine
```

**Solution** (requires code change):
The `from_dict()` method in `base_config.py` needs to recursively convert nested dictionaries back to their respective dataclass types using something like:
```python
@classmethod
def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
    """Create config from dictionary with proper nested conversion."""
    # Convert nested configs
    for key, value in config_dict.items():
        if key == "loss_config" and isinstance(value, dict):
            config_dict[key] = LossConfig(**value)
        elif key == "logging_config" and isinstance(value, dict):
            config_dict[key] = LoggingConfig(**value)
        # ... etc for other nested configs
    return cls(**config_dict)
```

---

### Issue 2: Composite Loss Tensor Handling

**Affected Test**:
- `test_composite_loss_forward`

**Error Message**:
```
RuntimeError: a Tensor with 50 elements cannot be converted to Scalar
```

**Root Cause**:
When individual losses return tensors with multiple elements (non-reduced losses), calling `.item()` on them fails. The code tries to convert a multi-element tensor to a scalar for tracking purposes.

**Location**: `loss_framework/core/composite_loss.py:146`

**Workaround**:
Use losses with `reduction='mean'` or `reduction='sum'` when using CompositeLoss:
```python
# ✅ Works - use reduced losses
composite = CompositeLoss({
    "mse": nn.MSELoss(reduction='mean'),
    "l1": nn.L1Loss(reduction='mean')
})

# ❌ Fails - non-reduced losses
composite = CompositeLoss({
    "mse": nn.MSELoss(reduction='none'),
    "l1": nn.L1Loss(reduction='none')
})
```

**Solution** (requires code change):
Modify line 146 in `composite_loss.py`:
```python
# Current (broken):
loss_value.item() if torch.is_tensor(loss_value) else loss_value

# Fixed:
if torch.is_tensor(loss_value):
    if loss_value.numel() == 1:
        self._current_losses[name] = loss_value.item()
    else:
        self._current_losses[name] = loss_value.mean().item()
else:
    self._current_losses[name] = loss_value
```

---

### Issue 3: Weight Schedule Assertion Mismatch

**Affected Test**:
- `test_weight_update`

**Error Message**:
```
assert 1.9026092617400676 == 1.0
```

**Root Cause**:
The test expects the weight to be exactly 1.0 at epoch 0, but with the exponential schedule, the weight starts increasing immediately. The default schedule type is "exponential", not the expected behavior.

**Workaround**:
When using AdaptiveWeightedLoss, explicitly set `schedule_type='linear'` for predictable behavior:
```python
# ✅ Predictable behavior
loss = AdaptiveWeightedLoss(
    schedule_type='linear',  # Specify explicitly
    warmup_epochs=5,
    initial_weight=1.0
)

# ❌ Default may vary
loss = AdaptiveWeightedLoss(warmup_epochs=5)  # Uses exponential by default
```

**Solution** (requires code change):
Update the test to either:
1. Use `schedule_type='constant'` for predictable testing
2. Check approximate values instead of exact equality
3. Update expected values for exponential schedule

---

### Issue 4: Device Mismatch in PhysicsInspiredLoss

**Affected Tests**:
- `test_physics_loss_forward`
- `test_conservation_loss`

**Error Message**:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**Root Cause**:
When the model runs on CUDA, the features tensor passed to the loss function remains on CPU, causing device mismatch during Hamiltonian computation.

**Workaround**:
Ensure all tensors are on the same device:
```python
# ✅ Move features to same device as predictions
predictions = model(inputs)  # On CUDA
features = features.to(predictions.device)  # Also on CUDA
loss = physics_loss(predictions, targets, features=features)
```

**Solution** (requires code change):
Add automatic device handling in `physics_inspired_loss.py`:
```python
def _compute_hamiltonian_loss(self, features):
    # Ensure features are on correct device
    if self.potential_net[0].weight.device != features.device:
        features = features.to(self.potential_net[0].weight.device)
    # ... rest of computation
```

---

### Issue 5: Exception Type Mismatch in Validators

**Affected Test**:
- `test_invalid_input_types`

**Error Message**:
Test expects `TypeError` but gets `ValueError`

**Root Cause**:
The `InputValidator.validate_shape()` method raises `ValueError` for shape mismatches, but the test expects `TypeError`. This is a test issue, not a code issue.

**Workaround**:
Users should catch `ValueError` for shape validation:
```python
try:
    InputValidator.validate_shape(tensor, expected_shape=(32, 5))
except ValueError as e:  # ✅ Correct exception type
    print(f"Shape error: {e}")
```

**Solution** (requires test change):
Update `test_integration.py` line 308:
```python
# Current (incorrect expectation):
with pytest.raises(TypeError):  # ❌ Wrong type

# Fixed:
with pytest.raises(ValueError):  # ✅ Correct type
    InputValidator.validate_shape(predictions, expected_shape=(32, 5))
```

---

## 🟡 Minor Issues

### Issue 6: Type Mismatches in Loss Config

**Files Affected**:
- `loss_framework/losses/adaptive_weighted_loss.py`
- Various config files

**Issues**:
- `np.floating` not assignable to `float` type hints
- Some functions return `Dict[str, np.floating]` instead of `Dict[str, float]`

**Impact**: 
Type checker warnings only - runtime works fine.

**Solution**: 
Add explicit type conversion:
```python
return {k: float(v) for k, v in stats.items()}
```

---

## ✅ Recommended Usage Patterns

### For Users (Workarounds):

1. **Avoid CompositeLoss with non-reduced losses** - Use `reduction='mean'`
2. **Always specify device explicitly** - Use `.to(device)` on all tensors
3. **Use direct config objects** - Avoid YAML serialization for now
4. **Use linear schedule** - More predictable than exponential for testing

### For Developers (Fixes):

1. Fix composite loss tensor handling
2. Implement proper nested config deserialization
3. Add automatic device synchronization
4. Update tests to match actual behavior

---

## 🧪 Testing Recommendations

### Running Tests

```bash
# Run all tests (expect 7 failures)
pytest loss_framework/tests/ -v

# Run only passing tests
pytest loss_framework/tests/ -v --ignore=loss_framework/tests/test_integration.py

# Run specific test files
pytest loss_framework/tests/test_novel_losses.py -v
pytest loss_framework/tests/test_core.py -v
```

### Expected Results

- `test_novel_losses.py`: 19/22 passing
- `test_core.py`: 10/11 passing
- `test_config.py`: 13/15 passing
- `test_integration.py`: 6/9 passing

---

## 📝 Version Info

**Framework Version**: 1.0.0
**Python Version**: 3.11.11
**PyTorch Version**: 2.5.1+ / 2.6.0+
**CUDA Version**: 11.8
**Platform**: Windows 10/11

---

## 🔄 Planned Fixes

Priority order for fixes:
1. **High**: Composite loss tensor handling
2. **High**: Config serialization
3. **Medium**: Device synchronization in PhysicsInspiredLoss
4. **Low**: Test expectation updates
5. **Low**: Type hint corrections

---

**Last Updated**: 2026-02-17
**Maintained by**: AI Development Team
