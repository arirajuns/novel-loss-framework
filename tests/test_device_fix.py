import torch
import sys

sys.path.insert(0, ".")

from loss_framework.losses import (
    InformationTheoreticLoss,
    AdaptiveWeightedLoss,
    RobustStatisticalLoss,
)

print("Testing device handling...")

# Test on available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create sample data on the device
predictions = torch.randn(32, 10, device=device)
targets = torch.randint(0, 10, (32,), device=device)

# Test InformationTheoreticLoss (the one that was failing)
print("\nTesting InformationTheoreticLoss...")
try:
    loss_fn = InformationTheoreticLoss()
    loss_val = loss_fn(predictions, targets)
    print(f"  SUCCESS: Loss = {loss_val.item():.4f}")
except Exception as e:
    print(f"  FAILED: {e}")

# Test AdaptiveWeightedLoss
print("\nTesting AdaptiveWeightedLoss...")
try:
    loss_fn = AdaptiveWeightedLoss()
    loss_val = loss_fn(predictions, targets)
    print(f"  SUCCESS: Loss = {loss_val.item():.4f}")
except Exception as e:
    print(f"  FAILED: {e}")

# Test RobustStatisticalLoss
print("\nTesting RobustStatisticalLoss...")
try:
    loss_fn = RobustStatisticalLoss(robust_type="huber")
    loss_val = loss_fn(predictions, targets)
    print(f"  SUCCESS: Loss = {loss_val.item():.4f}")
except Exception as e:
    print(f"  FAILED: {e}")

# Test standard CrossEntropy
print("\nTesting CrossEntropy (baseline)...")
try:
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_val = loss_fn(predictions, targets)
    print(f"  SUCCESS: Loss = {loss_val.item():.4f}")
except Exception as e:
    print(f"  FAILED: {e}")

print("\n" + "=" * 70)
print("Device handling test complete!")
print("=" * 70)
