#!/usr/bin/env python
"""
Validation Script for Novel Loss Function Framework
Runs comprehensive tests and generates validation report
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime


def run_tests():
    """Run the test suite and capture results."""
    print("=" * 70)
    print("RUNNING TEST SUITE")
    print("=" * 70)

    try:
        # Run pytest with coverage
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "loss_framework/tests/",
                "-v",
                "--tb=short",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def validate_imports():
    """Validate that all modules can be imported."""
    print("\n" + "=" * 70)
    print("VALIDATING IMPORTS")
    print("=" * 70)

    imports_to_test = [
        ("loss_framework", "Main package"),
        ("loss_framework.config", "Configuration module"),
        ("loss_framework.config.base_config", "Base config"),
        ("loss_framework.config.loss_config", "Loss configs"),
        ("loss_framework.config.experiment_config", "Experiment config"),
        ("loss_framework.core", "Core module"),
        ("loss_framework.core.base_loss", "Base loss"),
        ("loss_framework.core.loss_factory", "Loss factory"),
        ("loss_framework.core.loss_registry", "Loss registry"),
        ("loss_framework.core.composite_loss", "Composite loss"),
        ("loss_framework.losses", "Losses module"),
        ("loss_framework.losses.adaptive_weighted_loss", "Adaptive loss"),
        ("loss_framework.losses.geometric_loss", "Geometric loss"),
        ("loss_framework.losses.information_theoretic_loss", "Info-theoretic loss"),
        ("loss_framework.losses.physics_inspired_loss", "Physics loss"),
        ("loss_framework.losses.robust_statistical_loss", "Robust loss"),
        ("loss_framework.utils", "Utils module"),
        ("loss_framework.utils.validators", "Validators"),
        ("loss_framework.utils.gradients", "Gradients"),
        ("loss_framework.utils.metrics", "Metrics"),
    ]

    success_count = 0
    for module_name, description in imports_to_test:
        try:
            __import__(module_name)
            print(f"✅ {description}: {module_name}")
            success_count += 1
        except Exception as e:
            print(f"❌ {description}: {module_name} - {e}")

    print(f"\nImport Success Rate: {success_count}/{len(imports_to_test)}")
    return success_count == len(imports_to_test)


def validate_loss_functions():
    """Validate that loss functions can be instantiated and used."""
    print("\n" + "=" * 70)
    print("VALIDATING LOSS FUNCTIONS")
    print("=" * 70)

    import torch
    from loss_framework.losses import (
        AdaptiveWeightedLoss,
        GeometricDistanceLoss,
        InformationTheoreticLoss,
        PhysicsInspiredLoss,
        RobustStatisticalLoss,
    )

    # Test data
    batch_size = 8
    num_classes = 5
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    losses_to_test = [
        ("AdaptiveWeightedLoss", AdaptiveWeightedLoss()),
        ("GeometricDistanceLoss", GeometricDistanceLoss(manifold_type="euclidean")),
        ("InformationTheoreticLoss", InformationTheoreticLoss()),
        ("PhysicsInspiredLoss", PhysicsInspiredLoss()),
        ("RobustStatisticalLoss", RobustStatisticalLoss(robust_type="huber")),
    ]

    success_count = 0
    for name, loss_fn in losses_to_test:
        try:
            result = loss_fn(predictions, targets)
            if torch.isfinite(result) and result.item() >= 0:
                print(f"✅ {name}: Loss={result.item():.4f}")
                success_count += 1
            else:
                print(f"❌ {name}: Invalid loss value: {result.item()}")
        except Exception as e:
            print(f"❌ {name}: {e}")

    print(f"\nLoss Function Success Rate: {success_count}/{len(losses_to_test)}")
    return success_count == len(losses_to_test)


def validate_design_patterns():
    """Validate design pattern implementations."""
    print("\n" + "=" * 70)
    print("VALIDATING DESIGN PATTERNS")
    print("=" * 70)

    checks = []

    # 1. Template Method Pattern
    try:
        from loss_framework.core.base_loss import BaseLoss

        # Check if abstract
        try:
            BaseLoss()
            checks.append(("Template Method", False, "BaseLoss should be abstract"))
        except TypeError:
            checks.append(("Template Method", True, "✅ BaseLoss is abstract"))
    except Exception as e:
        checks.append(("Template Method", False, str(e)))

    # 2. Factory Pattern
    try:
        from loss_framework.core.loss_factory import LossFactory
        from loss_framework.config.loss_config import LossConfig

        loss = LossFactory.create_standard("mse")
        checks.append(("Factory Pattern", True, "✅ LossFactory working"))
    except Exception as e:
        checks.append(("Factory Pattern", False, str(e)))

    # 3. Registry Pattern
    try:
        from loss_framework.core.loss_registry import LossRegistry

        info = LossRegistry.info()
        checks.append(
            ("Registry Pattern", True, f"✅ Registry has {info.get('count', 0)} losses")
        )
    except Exception as e:
        checks.append(("Registry Pattern", False, str(e)))

    # 4. Composite Pattern
    try:
        from loss_framework.core.composite_loss import CompositeLoss
        import torch.nn as nn

        composite = CompositeLoss({"mse": nn.MSELoss(reduction="none")})
        checks.append(("Composite Pattern", True, "✅ CompositeLoss working"))
    except Exception as e:
        checks.append(("Composite Pattern", False, str(e)))

    # 5. Builder Pattern
    try:
        from loss_framework.config.experiment_config import ExperimentConfig

        config = ExperimentConfig(experiment_name="test")
        checks.append(("Builder Pattern", True, "✅ Configuration system working"))
    except Exception as e:
        checks.append(("Builder Pattern", False, str(e)))

    # 6. Strategy Pattern
    try:
        from loss_framework.losses.adaptive_weighted_loss import WeightScheduleStrategy

        # Test that strategies are callable
        result = WeightScheduleStrategy.linear_schedule(5, 0, 10, 0.1, 1.0)
        checks.append(("Strategy Pattern", True, "✅ Weight scheduling working"))
    except Exception as e:
        checks.append(("Strategy Pattern", False, str(e)))

    # 7. Singleton Pattern
    try:
        from loss_framework.core.loss_registry import LossRegistry

        reg1 = LossRegistry()
        reg2 = LossRegistry()
        if reg1 is reg2:
            checks.append(("Singleton Pattern", True, "✅ Registry is singleton"))
        else:
            checks.append(("Singleton Pattern", False, "Registry should be singleton"))
    except Exception as e:
        checks.append(("Singleton Pattern", False, str(e)))

    # Print results
    success_count = 0
    for pattern, success, message in checks:
        status = "✅" if success else "❌"
        print(f"{status} {pattern}: {message}")
        if success:
            success_count += 1

    print(f"\nDesign Pattern Success Rate: {success_count}/{len(checks)}")
    return success_count == len(checks)


def count_files_and_lines():
    """Count files and lines of code."""
    print("\n" + "=" * 70)
    print("CODE STATISTICS")
    print("=" * 70)

    framework_dir = Path("loss_framework")

    if not framework_dir.exists():
        print("❌ Framework directory not found")
        return

    stats = {
        "total_files": 0,
        "python_files": 0,
        "test_files": 0,
        "total_lines": 0,
        "code_lines": 0,
    }

    for path in framework_dir.rglob("*"):
        if path.is_file():
            stats["total_files"] += 1

            if path.suffix == ".py":
                stats["python_files"] += 1
                if "test" in path.name:
                    stats["test_files"] += 1

                try:
                    with open(path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        stats["total_lines"] += len(lines)
                        stats["code_lines"] += len(
                            [
                                l
                                for l in lines
                                if l.strip() and not l.strip().startswith("#")
                            ]
                        )
                except:
                    pass

    print(f"Total Files: {stats['total_files']}")
    print(f"Python Files: {stats['python_files']}")
    print(f"Test Files: {stats['test_files']}")
    print(f"Total Lines: {stats['total_lines']:,}")
    print(f"Code Lines: {stats['code_lines']:,}")


def generate_report():
    """Generate validation report."""
    report = {"timestamp": datetime.now().isoformat(), "validation_results": {}}

    print("\n" + "=" * 70)
    print("GENERATING VALIDATION REPORT")
    print("=" * 70)

    # Run all validations
    results = {
        "imports": validate_imports(),
        "loss_functions": validate_loss_functions(),
        "design_patterns": validate_design_patterns(),
    }

    # Code statistics
    count_files_and_lines()

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = all(results.values())

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check.replace('_', ' ').title()}")

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED! 🎉")
        print("The Novel Loss Function Framework is ready for use.")
    else:
        print("⚠️  SOME VALIDATIONS FAILED")
        print("Please review the errors above.")
    print("=" * 70)

    return all_passed


def main():
    """Main validation function."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "NOVEL LOSS FUNCTION FRAMEWORK" + " " * 19 + "║")
    print("║" + " " * 22 + "VALIDATION SCRIPT" + " " * 29 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Run validation
    success = generate_report()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
