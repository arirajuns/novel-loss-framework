"""
Benchmark Suite for Loss Function Comparison
Comprehensive comparison between novel and standard loss functions
Uses MNIST and synthetic datasets for evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import our framework
from loss_framework.core.loss_factory import LossFactory
from loss_framework.losses import (
    AdaptiveWeightedLoss,
    GeometricDistanceLoss,
    InformationTheoreticLoss,
    PhysicsInspiredLoss,
    RobustStatisticalLoss,
)
from loss_framework.utils.metrics import MetricsCalculator
from loss_framework.utils.gradients import GradientUtils


@dataclass
class BenchmarkResult:
    """Data class for benchmark results."""

    loss_name: str
    dataset: str
    final_loss: float
    final_accuracy: float
    training_time: float
    convergence_epoch: int
    loss_history: List[float]
    accuracy_history: List[float]
    gradient_norms: List[float]
    stability_score: float
    robustness_to_noise: float
    memory_usage_mb: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class LossFunctionBenchmark:
    """
    Benchmark suite for comparing loss functions.

    Compares:
    1. Training convergence speed
    2. Final accuracy/performance
    3. Gradient stability
    4. Robustness to noise
    5. Memory efficiency
    6. Computational overhead
    """

    def __init__(self, output_dir: str = "loss_framework/experiments/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []

    def get_mnist_data(self, train: bool = True, batch_size: int = 64) -> DataLoader:
        """Load MNIST dataset."""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        dataset = torchvision.datasets.MNIST(
            root="./data", train=train, download=True, transform=transform
        )

        return DataLoader(dataset, batch_size=batch_size, shuffle=train)

    def get_synthetic_data(
        self,
        n_samples: int = 1000,
        n_features: int = 20,
        n_classes: int = 5,
        noise_level: float = 0.1,
    ) -> Tuple[TensorDataset, TensorDataset]:
        """Generate synthetic classification data."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Generate data
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, n_classes, n_samples)

        # Add noise
        noise_mask = np.random.rand(n_samples) < noise_level
        y[noise_mask] = np.random.randint(0, n_classes, noise_mask.sum())

        # Split train/test
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        train_dataset = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

        return train_dataset, test_dataset

    def train_model(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 10,
        lr: float = 0.001,
    ) -> Dict[str, Any]:
        """Train model and collect metrics."""
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.to(self.device)

        loss_history = []
        accuracy_history = []
        gradient_norms = []

        start_time = time.time()

        for epoch in range(epochs):
            model.train()
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)

                # Handle different loss function signatures
                if isinstance(
                    loss_fn,
                    (
                        GeometricDistanceLoss,
                        InformationTheoreticLoss,
                        PhysicsInspiredLoss,
                    ),
                ):
                    # These might need features
                    loss = loss_fn(output, target)
                else:
                    loss = loss_fn(output, target)

                loss.backward()

                # Track gradient norm
                grad_norm = GradientUtils.compute_gradient_norm(model)
                gradient_norms.append(grad_norm)

                optimizer.step()

                epoch_losses.append(loss.item())

                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_total += target.size(0)

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            accuracy = 100.0 * epoch_correct / epoch_total

            loss_history.append(avg_loss)
            accuracy_history.append(accuracy)

            if epoch % 2 == 0:
                print(f"  Epoch {epoch}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")

        training_time = time.time() - start_time

        # Final evaluation on test set
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += target.size(0)

        final_accuracy = 100.0 * test_correct / test_total

        # Calculate convergence (loss < threshold)
        threshold = loss_history[0] * 0.1  # 90% reduction
        convergence_epoch = next(
            (i for i, l in enumerate(loss_history) if l < threshold), epochs
        )

        # Calculate stability (variance of gradient norms)
        stability_score = 1.0 / (1.0 + np.std(gradient_norms[-10:]))

        # Memory usage (approximate)
        memory_usage = (
            torch.cuda.memory_allocated(self.device) / (1024**2)
            if torch.cuda.is_available()
            else 0
        )

        return {
            "final_loss": loss_history[-1],
            "final_accuracy": final_accuracy,
            "training_time": training_time,
            "convergence_epoch": convergence_epoch,
            "loss_history": loss_history,
            "accuracy_history": accuracy_history,
            "gradient_norms": gradient_norms,
            "stability_score": stability_score,
            "memory_usage_mb": memory_usage,
        }

    def benchmark_loss_functions(
        self, dataset_type: str = "mnist", epochs: int = 10
    ) -> List[BenchmarkResult]:
        """
        Benchmark all loss functions.

        Args:
            dataset_type: 'mnist' or 'synthetic'
            epochs: Number of training epochs

        Returns:
            List of benchmark results
        """
        print(f"\n{'=' * 70}")
        print(f"BENCHMARKING LOSS FUNCTIONS ON {dataset_type.upper()}")
        print(f"{'=' * 70}\n")

        # Load data
        if dataset_type == "mnist":
            train_loader = self.get_mnist_data(train=True)
            test_loader = self.get_mnist_data(train=False)
        else:
            train_data, test_data = self.get_synthetic_data()
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # Define loss functions to compare
        loss_functions = [
            ("CrossEntropy (Standard)", nn.CrossEntropyLoss()),
            ("MSE (Standard)", nn.MSELoss()),
            ("AdaptiveWeighted (Ours)", AdaptiveWeightedLoss()),
            ("InformationTheoretic (Ours)", InformationTheoreticLoss()),
            (
                "RobustStatistical-Huber (Ours)",
                RobustStatisticalLoss(robust_type="huber"),
            ),
            (
                "RobustStatistical-Tukey (Ours)",
                RobustStatisticalLoss(robust_type="tukey"),
            ),
        ]

        results = []

        for loss_name, loss_fn in loss_functions:
            print(f"\nTesting: {loss_name}")
            print("-" * 70)

            # Create fresh model
            if dataset_type == "mnist":
                model = SimpleCNN()
            else:
                model = nn.Sequential(nn.Linear(20, 50), nn.ReLU(), nn.Linear(50, 5))

            # Train and collect metrics
            metrics = self.train_model(
                model, loss_fn, train_loader, test_loader, epochs
            )

            # Robustness test (with noise)
            if dataset_type == "synthetic":
                train_data_noisy, test_data_noisy = self.get_synthetic_data(
                    noise_level=0.3
                )
                train_loader_noisy = DataLoader(
                    train_data_noisy, batch_size=32, shuffle=True
                )
                test_loader_noisy = DataLoader(
                    test_data_noisy, batch_size=32, shuffle=False
                )

                model_robust = nn.Sequential(
                    nn.Linear(20, 50), nn.ReLU(), nn.Linear(50, 5)
                )
                metrics_noisy = self.train_model(
                    model_robust,
                    loss_fn,
                    train_loader_noisy,
                    test_loader_noisy,
                    epochs=5,
                )
                robustness = metrics_noisy["final_accuracy"] / metrics["final_accuracy"]
            else:
                robustness = 1.0

            result = BenchmarkResult(
                loss_name=loss_name,
                dataset=dataset_type,
                final_loss=metrics["final_loss"],
                final_accuracy=metrics["final_accuracy"],
                training_time=metrics["training_time"],
                convergence_epoch=metrics["convergence_epoch"],
                loss_history=metrics["loss_history"],
                accuracy_history=metrics["accuracy_history"],
                gradient_norms=metrics["gradient_norms"],
                stability_score=metrics["stability_score"],
                robustness_to_noise=robustness,
                memory_usage_mb=metrics["memory_usage_mb"],
            )

            results.append(result)

        self.results.extend(results)
        return results

    def save_results(self, filename: str = None):
        """Save benchmark results to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        filepath = self.output_dir / filename

        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "results": [r.to_dict() for r in self.results],
        }

        with open(filepath, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to: {filepath}")
        return filepath

    def generate_report(self) -> str:
        """Generate comprehensive comparison report."""
        if not self.results:
            return "No results to report"

        report = []
        report.append("=" * 70)
        report.append("LOSS FUNCTION BENCHMARK REPORT")
        report.append("=" * 70)
        report.append("")

        # Group by dataset
        datasets = set(r.dataset for r in self.results)

        for dataset in datasets:
            dataset_results = [r for r in self.results if r.dataset == dataset]

            report.append(f"\nDataset: {dataset.upper()}")
            report.append("-" * 70)

            # Sort by final accuracy
            dataset_results.sort(key=lambda x: x.final_accuracy, reverse=True)

            report.append(
                f"{'Rank':<6}{'Loss Function':<35}{'Final Acc':<12}{'Time (s)':<12}{'Convergence':<12}"
            )
            report.append("-" * 70)

            for i, r in enumerate(dataset_results, 1):
                report.append(
                    f"{i:<6}{r.loss_name:<35}{r.final_accuracy:<12.2f}{r.training_time:<12.2f}{r.convergence_epoch:<12}"
                )

            report.append("")

            # Best performers
            best_accuracy = max(dataset_results, key=lambda x: x.final_accuracy)
            fastest = min(dataset_results, key=lambda x: x.training_time)
            most_stable = max(dataset_results, key=lambda x: x.stability_score)

            report.append("Best Performers:")
            report.append(
                f"  🏆 Best Accuracy: {best_accuracy.loss_name} ({best_accuracy.final_accuracy:.2f}%)"
            )
            report.append(
                f"  ⚡ Fastest Training: {fastest.loss_name} ({fastest.training_time:.2f}s)"
            )
            report.append(
                f"  🎯 Most Stable: {most_stable.loss_name} (score: {most_stable.stability_score:.3f})"
            )

            # Novel vs Standard comparison
            standard = [r for r in dataset_results if "Standard" in r.loss_name]
            novel = [r for r in dataset_results if "Ours" in r.loss_name]

            if standard and novel:
                avg_acc_standard = np.mean([r.final_accuracy for r in standard])
                avg_acc_novel = np.mean([r.final_accuracy for r in novel])

                report.append(f"\n📊 Average Performance:")
                report.append(f"   Standard Losses: {avg_acc_standard:.2f}%")
                report.append(f"   Novel Losses (Ours): {avg_acc_novel:.2f}%")
                report.append(
                    f"   Improvement: {avg_acc_novel - avg_acc_standard:+.2f}%"
                )

        report.append("\n" + "=" * 70)

        report_text = "\n".join(report)

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"benchmark_report_{timestamp}.txt"
        with open(report_file, "w") as f:
            f.write(report_text)

        print(report_text)
        print(f"\nReport saved to: {report_file}")

        return report_text


def run_benchmark_comparison():
    """Run complete benchmark comparison."""
    benchmark = LossFunctionBenchmark()

    # Benchmark on MNIST
    print("\n" + "=" * 70)
    print("STARTING COMPREHENSIVE LOSS FUNCTION BENCHMARK")
    print("=" * 70)

    results_mnist = benchmark.benchmark_loss_functions(dataset_type="mnist", epochs=10)

    # Benchmark on synthetic data
    results_synthetic = benchmark.benchmark_loss_functions(
        dataset_type="synthetic", epochs=10
    )

    # Save results
    benchmark.save_results()

    # Generate report
    report = benchmark.generate_report()

    return benchmark, report


if __name__ == "__main__":
    benchmark, report = run_benchmark_comparison()
