"""
Main execution module for running experiments and tests
Provides CLI interface and experiment orchestration
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional
import torch

from loss_framework.config.experiment_config import ExperimentConfig
from loss_framework.core.loss_factory import LossFactory


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("loss_framework.log"),
        ],
    )
    return logging.getLogger(__name__)


def run_tests(test_path: Optional[str] = None, verbose: bool = False) -> int:
    """
    Run the test suite.

    Args:
        test_path: Specific test path to run (None for all)
        verbose: Whether to run in verbose mode

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import pytest

    args = [test_path or "loss_framework/tests/"]
    if verbose:
        args.append("-v")
    args.append("--tb=short")

    logger = logging.getLogger(__name__)
    logger.info(f"Running tests with args: {args}")

    exit_code = pytest.main(args)
    return exit_code


def run_experiment(config_path: str) -> None:
    """
    Run an experiment from configuration.

    Args:
        config_path: Path to experiment configuration file
    """
    logger = logging.getLogger(__name__)

    # Load configuration
    config = ExperimentConfig.load(config_path)
    logger.info(f"Loaded experiment config: {config.experiment_name}")

    # Create loss function
    loss_fn = LossFactory.create_from_config(config.loss_config)
    logger.info(f"Created loss function: {loss_fn}")

    # Save configuration to experiment directory
    config.save()
    logger.info(f"Saved configuration to: {config.get_experiment_dir()}")

    # Note: Full training loop would be implemented here
    logger.info("Experiment setup complete. Training loop not implemented in demo.")


def list_losses() -> None:
    """List all available loss functions."""
    from loss_framework.core.loss_registry import LossRegistry

    print("\n=== Available Loss Functions ===\n")

    # Standard losses
    standard = LossFactory.list_available_losses()["standard"]
    print("Standard PyTorch Losses:")
    for loss in standard:
        print(f"  - {loss}")

    # Registered losses
    registered = LossRegistry.list_losses()
    if registered:
        print("\nRegistered Custom Losses:")
        for loss in registered:
            metadata = LossRegistry.get_metadata(loss)
            category = (
                metadata.get("category", "uncategorized")
                if metadata
                else "uncategorized"
            )
            print(f"  - {loss} ({category})")

    # Categories
    categories = LossRegistry.get_categories()
    if categories:
        print(f"\nCategories: {', '.join(categories)}")

    print()


def demo_loss_functions() -> None:
    """Run a demo of loss functions."""
    print("\n=== Novel Loss Function Demo ===\n")

    # Create sample data
    predictions = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))

    # Test each loss
    losses_to_test = [
        (
            "AdaptiveWeightedLoss",
            lambda: LossFactory.create_from_config(
                ExperimentConfig.__dataclass_fields__["loss_config"].default_factory()
            ),
        ),
    ]

    print("Testing loss functions with sample data...")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}\n")

    # Note: In full implementation, would test all losses
    print("Demo complete. See tests for full loss function demos.\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Novel Loss Function Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python -m loss_framework main --test
  
  # Run specific test file
  python -m loss_framework main --test --test-path loss_framework/tests/test_core.py
  
  # List available losses
  python -m loss_framework main --list-losses
  
  # Run experiment
  python -m loss_framework main --experiment config.yaml
  
  # Run demo
  python -m loss_framework main --demo
        """,
    )

    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument(
        "--test-path", type=str, default=None, help="Specific test path to run"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--experiment",
        type=str,
        metavar="CONFIG",
        help="Run experiment from config file",
    )
    parser.add_argument(
        "--list-losses", action="store_true", help="List available loss functions"
    )
    parser.add_argument("--demo", action="store_true", help="Run loss function demo")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting Novel Loss Function Framework")

    # Handle commands
    if args.test:
        exit_code = run_tests(args.test_path, args.verbose)
        sys.exit(exit_code)

    elif args.experiment:
        run_experiment(args.experiment)

    elif args.list_losses:
        list_losses()

    elif args.demo:
        demo_loss_functions()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
