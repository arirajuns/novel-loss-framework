"""
Quick Hugging Face Dataset Test
Simplified test comparing losses on real datasets
"""

import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
import json
from datetime import datetime

# Import our framework
import sys

sys.path.insert(0, ".")
from loss_framework.losses import (
    AdaptiveWeightedLoss,
    InformationTheoreticLoss,
    RobustStatisticalLoss,
)


def test_on_imdb_sample():
    """Quick test on IMDB sample."""
    print("=" * 70)
    print("HUGGING FACE DATASET TEST: IMDB Sample")
    print("=" * 70)

    # Load small sample
    print("\nLoading IMDB dataset...")
    dataset = load_dataset("imdb", split="train[:1000]")  # Small sample for speed
    test_dataset = load_dataset("imdb", split="test[:200]")

    print(f"Train samples: {len(dataset)}, Test samples: {len(test_dataset)}")

    # Simple feature extraction (bag of words)
    print("\nPreparing features...")
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(max_features=1000, stop_words="english")
    X_train = vectorizer.fit_transform(dataset["text"]).toarray()
    y_train = np.array(dataset["label"])
    X_test = vectorizer.transform(test_dataset["text"]).toarray()
    y_test = np.array(test_dataset["label"])

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    print(f"Feature dimension: {X_train.shape[1]}")

    # Simple model
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    # Test each loss function
    losses_to_test = [
        ("CrossEntropy (PyTorch)", nn.CrossEntropyLoss(), "pytorch"),
        ("AdaptiveWeighted (Ours)", AdaptiveWeightedLoss(), "novel"),
        ("InformationTheoretic (Ours)", InformationTheoreticLoss(), "novel"),
        (
            "RobustStatistical-Huber (Ours)",
            RobustStatisticalLoss(robust_type="huber"),
            "novel",
        ),
    ]

    results = []

    print("\n" + "=" * 70)
    print("TRAINING & EVALUATION")
    print("=" * 70)

    for loss_name, loss_fn, category in losses_to_test:
        print(f"\n{'-' * 70}")
        print(f"Testing: {loss_name}")
        print(f"{'-' * 70}")

        # Create model
        model = SimpleClassifier(X_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training
        epochs = 5
        batch_size = 32
        n_batches = len(X_train) // batch_size

        train_losses = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0

            # Shuffle
            perm = torch.randperm(len(X_train))
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size

                batch_x = X_shuffled[start:end]
                batch_y = y_shuffled[start:end]

                optimizer.zero_grad()
                output = model(batch_x)
                loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / n_batches
            train_losses.append(avg_loss)

            print(f"  Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_output = model(X_test)
            test_pred = test_output.argmax(dim=1)
            test_acc = (test_pred == y_test).float().mean().item() * 100

        print(f"  Test Accuracy: {test_acc:.2f}%")

        results.append(
            {
                "loss_name": loss_name,
                "category": category,
                "final_loss": train_losses[-1],
                "test_accuracy": test_acc,
                "loss_history": train_losses,
            }
        )

    # Generate report
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    # Sort by accuracy
    results.sort(key=lambda x: x["test_accuracy"], reverse=True)

    print(f"{'Rank':<6}{'Loss Function':<40}{'Test Acc':<12}{'Final Loss':<12}")
    print("-" * 70)

    for i, r in enumerate(results, 1):
        marker = "✨" if r["category"] == "novel" else "  "
        print(
            f"{marker}{i:<4}{r['loss_name']:<40}{r['test_accuracy']:<12.2f}{r['final_loss']:<12.4f}"
        )

    # Compare categories
    pytorch_results = [r for r in results if r["category"] == "pytorch"]
    novel_results = [r for r in results if r["category"] == "novel"]

    if pytorch_results and novel_results:
        avg_pytorch = np.mean([r["test_accuracy"] for r in pytorch_results])
        avg_novel = np.mean([r["test_accuracy"] for r in novel_results])

        print("\n" + "=" * 70)
        print("CATEGORY COMPARISON")
        print("=" * 70)
        print(f"PyTorch (avg):   {avg_pytorch:.2f}%")
        print(f"Novel (avg):     {avg_novel:.2f}%")
        print(f"Difference:      {avg_novel - avg_pytorch:+.2f}%")
        print()

        if avg_novel > avg_pytorch:
            print("✅ CONCLUSION: Novel losses provide BETTER results on this dataset!")
            print(f"   Improvement: {avg_novel - avg_pytorch:.2f}%")
        elif avg_novel < avg_pytorch:
            print("⚠️  CONCLUSION: PyTorch losses perform better on this dataset")
            print(f"   Difference: {avg_pytorch - avg_novel:.2f}%")
        else:
            print("⚡ CONCLUSION: Performance is comparable")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = {
        "timestamp": timestamp,
        "dataset": "IMDB (sample)",
        "train_samples": len(dataset),
        "test_samples": len(test_dataset),
        "results": results,
    }

    with open(f"hf_test_results_{timestamp}.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: hf_test_results_{timestamp}.json")

    return results


if __name__ == "__main__":
    results = test_on_imdb_sample()
