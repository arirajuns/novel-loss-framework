"""
Hugging Face Dataset Testing Framework
Compares Novel Loss Functions with PyTorch Built-in Losses on Real Datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import time
import warnings

warnings.filterwarnings("ignore")

# Try to import Hugging Face datasets
try:
    from datasets import load_dataset

    HF_AVAILABLE = True
except ImportError:
    print("Hugging Face datasets not installed. Installing...")
    import subprocess

    subprocess.check_call(["pip", "install", "-q", "datasets"])
    from datasets import load_dataset

    HF_AVAILABLE = True

# Import our framework
import sys

sys.path.insert(0, ".")
from loss_framework.losses import (
    AdaptiveWeightedLoss,
    InformationTheoreticLoss,
    RobustStatisticalLoss,
    GeometricDistanceLoss,
)
from loss_framework.core.loss_factory import LossFactory
from loss_framework.utils.metrics import MetricsCalculator


@dataclass
class TestResult:
    """Results from a single test run."""

    dataset_name: str
    loss_name: str
    loss_category: str  # 'pytorch' or 'novel'
    model_type: str
    epochs: int
    final_train_loss: float
    final_val_loss: float
    final_val_accuracy: float
    best_val_accuracy: float
    training_time: float
    convergence_epoch: int
    loss_history: List[float]
    accuracy_history: List[float]
    val_loss_history: List[float]
    val_accuracy_history: List[float]
    gradient_stats: Dict[str, float]
    memory_usage_mb: float


class SimpleTextClassifier(nn.Module):
    """Simple text classifier for Hugging Face datasets."""

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.3
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Use last hidden state
        hidden = hidden[-1]  # (batch, hidden_dim)
        x = torch.relu(self.fc1(hidden))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SimpleIMDBClassifier(nn.Module):
    """Even simpler classifier for quick testing."""

    def __init__(
        self, vocab_size: int = 10000, embed_dim: int = 100, num_classes: int = 2
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        # Average pooling over sequence
        embedded = embedded.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        pooled = self.avg_pool(embedded).squeeze(-1)  # (batch, embed_dim)
        x = torch.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class HuggingFaceLossTester:
    """Test loss functions on Hugging Face datasets."""

    def __init__(self, output_dir: str = "loss_framework/experiments/hf_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []

    def load_hf_dataset(self, dataset_name: str = "imdb", max_samples: int = 5000):
        """Load and preprocess Hugging Face dataset."""
        print(f"Loading dataset: {dataset_name}")

        try:
            if dataset_name == "imdb":
                dataset = load_dataset("imdb", split="train")
                test_dataset = load_dataset("imdb", split="test")
            elif dataset_name == "sst2":
                dataset = load_dataset("glue", "sst2", split="train")
                test_dataset = load_dataset("glue", "sst2", split="validation")
            elif dataset_name == "ag_news":
                dataset = load_dataset("ag_news", split="train")
                test_dataset = load_dataset("ag_news", split="test")
            else:
                dataset = load_dataset(dataset_name, split="train")
                test_dataset = load_dataset(dataset_name, split="test")

            # Limit samples for faster testing
            if max_samples:
                dataset = dataset.shuffle(seed=42).select(
                    range(min(max_samples, len(dataset)))
                )
                test_dataset = test_dataset.shuffle(seed=42).select(
                    range(min(max_samples // 5, len(test_dataset)))
                )

            print(f"Train samples: {len(dataset)}, Test samples: {len(test_dataset)}")
            return dataset, test_dataset

        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return None, None

    def create_vocab_and_encode(self, texts, max_vocab_size=10000):
        """Create vocabulary and encode texts."""
        from collections import Counter

        # Tokenize (simple word split)
        all_words = []
        for text in texts:
            all_words.extend(str(text).lower().split())

        # Build vocab
        word_counts = Counter(all_words)
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, count in word_counts.most_common(max_vocab_size - 2):
            vocab[word] = len(vocab)

        return vocab

    def encode_text(self, text, vocab, max_len=256):
        """Encode single text."""
        words = str(text).lower().split()[:max_len]
        encoded = [vocab.get(word, vocab["<UNK>"]) for word in words]
        # Pad
        encoded = encoded + [vocab["<PAD>"]] * (max_len - len(encoded))
        return encoded

    def prepare_data(self, dataset, test_dataset, max_len=256):
        """Prepare data for PyTorch."""
        print("Preparing data...")

        # Create vocabulary from train data
        if hasattr(dataset, "text"):
            texts = dataset["text"]
        elif hasattr(dataset, "sentence"):
            texts = dataset["sentence"]
        else:
            texts = [str(item) for item in dataset]

        vocab = self.create_vocab_and_encode(texts)
        vocab_size = len(vocab)

        # Encode train data
        if hasattr(dataset, "text"):
            train_texts = dataset["text"]
            train_labels = dataset["label"]
        elif hasattr(dataset, "sentence"):
            train_texts = dataset["sentence"]
            train_labels = dataset["label"]

        train_encoded = [self.encode_text(text, vocab, max_len) for text in train_texts]
        train_labels = torch.tensor(train_labels)
        train_data = torch.tensor(train_encoded)

        # Encode test data
        if hasattr(test_dataset, "text"):
            test_texts = test_dataset["text"]
            test_labels = test_dataset["label"]
        elif hasattr(test_dataset, "sentence"):
            test_texts = test_dataset["sentence"]
            test_labels = test_dataset["label"]

        test_encoded = [self.encode_text(text, vocab, max_len) for text in test_texts]
        test_labels = torch.tensor(test_labels)
        test_data = torch.tensor(test_encoded)

        # Create dataloaders
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, test_loader, vocab_size

    def train_model(
        self, model, loss_fn, train_loader, test_loader, epochs=5, lr=0.001
    ) -> Dict[str, Any]:
        """Train model and collect metrics."""
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.to(self.device)

        loss_history = []
        accuracy_history = []
        val_loss_history = []
        val_accuracy_history = []

        start_time = time.time()
        best_val_acc = 0
        convergence_epoch = epochs

        for epoch in range(epochs):
            model.train()
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)

                loss = loss_fn(output, target)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_losses.append(loss.item())

                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_total += target.size(0)

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            train_acc = 100.0 * epoch_correct / epoch_total

            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    val_loss += loss_fn(output, target).item()

                    pred = output.argmax(dim=1, keepdim=True)
                    val_correct += pred.eq(target.view_as(pred)).sum().item()
                    val_total += target.size(0)

            avg_val_loss = val_loss / len(test_loader)
            val_acc = 100.0 * val_correct / val_total

            loss_history.append(avg_loss)
            accuracy_history.append(train_acc)
            val_loss_history.append(avg_val_loss)
            val_accuracy_history.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            # Check convergence (no improvement for 2 epochs)
            if epoch > 2 and val_acc < val_accuracy_history[-2] - 0.1:
                if convergence_epoch == epochs:
                    convergence_epoch = epoch

            if epoch % 1 == 0:
                print(
                    f"  Epoch {epoch}: Train Loss={avg_loss:.4f}, "
                    f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%"
                )

        training_time = time.time() - start_time

        # Final test accuracy
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

        final_test_acc = 100.0 * test_correct / test_total

        # Memory usage
        memory_usage = (
            torch.cuda.memory_allocated(self.device) / (1024**2)
            if torch.cuda.is_available()
            else 0
        )

        return {
            "final_train_loss": loss_history[-1],
            "final_val_loss": val_loss_history[-1],
            "final_val_accuracy": val_accuracy_history[-1],
            "best_val_accuracy": best_val_acc,
            "final_test_accuracy": final_test_acc,
            "training_time": training_time,
            "convergence_epoch": convergence_epoch,
            "loss_history": loss_history,
            "accuracy_history": accuracy_history,
            "val_loss_history": val_loss_history,
            "val_accuracy_history": val_accuracy_history,
            "memory_usage_mb": memory_usage,
        }

    def test_on_dataset(
        self, dataset_name: str = "imdb", epochs: int = 5
    ) -> List[TestResult]:
        """Test all loss functions on a specific dataset."""
        print(f"\n{'=' * 70}")
        print(f"TESTING ON {dataset_name.upper()} DATASET")
        print(f"{'=' * 70}\n")

        # Load dataset
        dataset, test_dataset = self.load_hf_dataset(dataset_name)
        if dataset is None:
            return []

        # Prepare data
        train_loader, test_loader, vocab_size = self.prepare_data(dataset, test_dataset)

        # Define loss functions to compare
        loss_functions = [
            ("CrossEntropy (PyTorch)", nn.CrossEntropyLoss(), "pytorch"),
            (
                "AdaptiveWeighted (Ours)",
                AdaptiveWeightedLoss(reduction="mean"),
                "novel",
            ),
            (
                "InformationTheoretic (Ours)",
                InformationTheoreticLoss(
                    use_entropy_regularization=True, entropy_weight=0.1
                ),
                "novel",
            ),
            (
                "RobustStatistical-Huber (Ours)",
                RobustStatisticalLoss(robust_type="huber", adaptive_scale=True),
                "novel",
            ),
        ]

        results = []

        for loss_name, loss_fn, category in loss_functions:
            print(f"\n{'-' * 70}")
            print(f"Testing: {loss_name}")
            print(f"{'-' * 70}")

            # Create fresh model
            model = SimpleIMDBClassifier(vocab_size=vocab_size, num_classes=2)

            # Train and evaluate
            try:
                metrics = self.train_model(
                    model, loss_fn, train_loader, test_loader, epochs
                )

                result = TestResult(
                    dataset_name=dataset_name,
                    loss_name=loss_name,
                    loss_category=category,
                    model_type="SimpleIMDBClassifier",
                    epochs=epochs,
                    final_train_loss=metrics["final_train_loss"],
                    final_val_loss=metrics["final_val_loss"],
                    final_val_accuracy=metrics["final_val_accuracy"],
                    best_val_accuracy=metrics["best_val_accuracy"],
                    training_time=metrics["training_time"],
                    convergence_epoch=metrics["convergence_epoch"],
                    loss_history=metrics["loss_history"],
                    accuracy_history=metrics["accuracy_history"],
                    val_loss_history=metrics["val_loss_history"],
                    val_accuracy_history=metrics["val_accuracy_history"],
                    gradient_stats={},
                    memory_usage_mb=metrics["memory_usage_mb"],
                )

                results.append(result)

                print(f"\n✅ Results for {loss_name}:")
                print(f"   Final Val Accuracy: {metrics['final_val_accuracy']:.2f}%")
                print(f"   Best Val Accuracy: {metrics['best_val_accuracy']:.2f}%")
                print(f"   Training Time: {metrics['training_time']:.2f}s")

            except Exception as e:
                print(f"❌ Error testing {loss_name}: {e}")
                import traceback

                traceback.print_exc()

        self.results.extend(results)
        return results

    def generate_report(self) -> str:
        """Generate comprehensive comparison report."""
        if not self.results:
            return "No results to report"

        report = []
        report.append("=" * 80)
        report.append("HUGGING FACE DATASET TESTING RESULTS")
        report.append("Novel Loss Functions vs PyTorch Built-in Losses")
        report.append("=" * 80)
        report.append("")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Group by dataset
        datasets = set(r.dataset_name for r in self.results)

        for dataset in datasets:
            dataset_results = [r for r in self.results if r.dataset_name == dataset]

            report.append(f"\n{'=' * 80}")
            report.append(f"DATASET: {dataset.upper()}")
            report.append(f"{'=' * 80}")
            report.append("")

            # Sort by final validation accuracy
            dataset_results.sort(key=lambda x: x.final_val_accuracy, reverse=True)

            # Results table
            report.append(
                f"{'Rank':<6}{'Loss Function':<40}{'Val Acc':<12}{'Best Acc':<12}{'Time':<12}"
            )
            report.append("-" * 80)

            for i, r in enumerate(dataset_results, 1):
                marker = "✨" if r.loss_category == "novel" else "  "
                report.append(
                    f"{marker}{i:<4}{r.loss_name:<40}{r.final_val_accuracy:<12.2f}"
                    f"{r.best_val_accuracy:<12.2f}{r.training_time:<12.2f}s"
                )

            report.append("")

            # Statistical comparison
            pytorch_results = [
                r for r in dataset_results if r.loss_category == "pytorch"
            ]
            novel_results = [r for r in dataset_results if r.loss_category == "novel"]

            if pytorch_results and novel_results:
                avg_pytorch = np.mean([r.final_val_accuracy for r in pytorch_results])
                avg_novel = np.mean([r.final_val_accuracy for r in novel_results])

                report.append("Category Comparison:")
                report.append(f"  PyTorch (avg): {avg_pytorch:.2f}%")
                report.append(f"  Novel (avg):   {avg_novel:.2f}%")
                report.append(f"  Difference:    {avg_novel - avg_pytorch:+.2f}%")

                if avg_novel > avg_pytorch:
                    report.append(
                        f"  ✅ Novel losses perform better by {avg_novel - avg_pytorch:.2f}%"
                    )
                elif avg_novel < avg_pytorch:
                    report.append(
                        f"  ⚠️  PyTorch losses perform better by {avg_pytorch - avg_novel:.2f}%"
                    )
                else:
                    report.append("  ⚡ Performance is similar")

                report.append("")

            # Best performer
            best = max(dataset_results, key=lambda x: x.final_val_accuracy)
            report.append(f"🏆 Best Performer: {best.loss_name}")
            report.append(f"   Validation Accuracy: {best.final_val_accuracy:.2f}%")
            report.append(
                f"   Category: {'Novel' if best.loss_category == 'novel' else 'PyTorch'}"
            )
            report.append("")

        # Overall summary
        report.append("=" * 80)
        report.append("OVERALL SUMMARY")
        report.append("=" * 80)
        report.append("")

        pytorch_all = [r for r in self.results if r.loss_category == "pytorch"]
        novel_all = [r for r in self.results if r.loss_category == "novel"]

        if pytorch_all and novel_all:
            overall_pytorch = np.mean([r.final_val_accuracy for r in pytorch_all])
            overall_novel = np.mean([r.final_val_accuracy for r in novel_all])

            report.append(f"PyTorch Losses (Overall): {overall_pytorch:.2f}%")
            report.append(f"Novel Losses (Overall):   {overall_novel:.2f}%")
            report.append(
                f"Difference:               {overall_novel - overall_pytorch:+.2f}%"
            )
            report.append("")

            if overall_novel > overall_pytorch:
                report.append(
                    "✅ CONCLUSION: Novel loss functions provide BETTER results!"
                )
                report.append(
                    f"   Average improvement: {overall_novel - overall_pytorch:.2f}%"
                )
            elif overall_novel < overall_pytorch:
                report.append(
                    "⚠️  CONCLUSION: PyTorch losses perform better on these datasets"
                )
                report.append(
                    f"   Average difference: {overall_pytorch - overall_novel:.2f}%"
                )
            else:
                report.append(
                    "⚡ CONCLUSION: Performance is comparable between categories"
                )

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"hf_test_results_{timestamp}.json"

        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "results": [asdict(r) for r in self.results],
        }

        with open(filepath, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to: {filepath}")
        return filepath


def run_hf_comparison():
    """Run complete Hugging Face dataset comparison."""
    print("=" * 80)
    print("HUGGING FACE DATASET COMPARISON")
    print("Testing Novel Loss Functions vs PyTorch Built-in")
    print("=" * 80)
    print()

    tester = HuggingFaceLossTester()

    # Test on IMDB
    try:
        tester.test_on_dataset("imdb", epochs=3)
    except Exception as e:
        print(f"Error testing IMDB: {e}")

    # Generate and print report
    report = tester.generate_report()
    print("\n" + report)

    # Save results
    tester.save_results()

    # Save report
    report_file = (
        tester.output_dir
        / f"hf_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(report_file, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {report_file}")

    return tester


if __name__ == "__main__":
    tester = run_hf_comparison()
