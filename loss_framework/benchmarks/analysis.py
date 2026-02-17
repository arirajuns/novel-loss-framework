"""
Analysis and Insights Module
Generates detailed analysis, visualizations, and insights from benchmark results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd


class BenchmarkAnalyzer:
    """
    Analyzer for benchmark results.
    Generates insights, comparisons, and recommendations.
    """

    def __init__(self, results_file: str = None):
        """Initialize analyzer with results file."""
        if results_file:
            with open(results_file, "r") as f:
                data = json.load(f)
                self.results = data.get("results", [])
                self.timestamp = data.get("timestamp", datetime.now().isoformat())
        else:
            self.results = []
            self.timestamp = datetime.now().isoformat()

        self.output_dir = Path("loss_framework/experiments/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_comparison_table(self) -> pd.DataFrame:
        """Create pandas DataFrame for easy comparison."""
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame(self.results)

        # Add category column
        df["category"] = df["loss_name"].apply(
            lambda x: "Novel (Ours)" if "Ours" in x else "Standard"
        )

        return df

    def statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis on results."""
        df = self.create_comparison_table()

        if df.empty:
            return {}

        analysis = {}

        # Group by category
        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset]

            analysis[dataset] = {}

            for category in ["Standard", "Novel (Ours)"]:
                cat_df = dataset_df[dataset_df["category"] == category]

                if not cat_df.empty:
                    analysis[dataset][category] = {
                        "count": len(cat_df),
                        "mean_accuracy": cat_df["final_accuracy"].mean(),
                        "std_accuracy": cat_df["final_accuracy"].std(),
                        "mean_time": cat_df["training_time"].mean(),
                        "mean_stability": cat_df["stability_score"].mean(),
                        "mean_convergence": cat_df["convergence_epoch"].mean(),
                        "best_accuracy": cat_df["final_accuracy"].max(),
                        "worst_accuracy": cat_df["final_accuracy"].min(),
                    }

        return analysis

    def generate_insights(self) -> List[str]:
        """Generate actionable insights from results."""
        df = self.create_comparison_table()

        if df.empty:
            return ["No results available for analysis"]

        insights = []

        # Overall performance comparison
        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset]

            standard_acc = dataset_df[dataset_df["category"] == "Standard"][
                "final_accuracy"
            ].mean()
            novel_acc = dataset_df[dataset_df["category"] == "Novel (Ours)"][
                "final_accuracy"
            ].mean()

            improvement = novel_acc - standard_acc

            if improvement > 0:
                insights.append(
                    f"✅ On {dataset.upper()}: Novel losses outperform standard by {improvement:+.2f}%"
                )
            else:
                insights.append(
                    f"⚠️  On {dataset.upper()}: Standard losses perform {abs(improvement):.2f}% better"
                )

        # Best overall performer
        best = df.loc[df["final_accuracy"].idxmax()]
        insights.append(
            f"🏆 Best overall: {best['loss_name']} with {best['final_accuracy']:.2f}% accuracy"
        )

        # Fastest convergence
        fastest = df.loc[df["convergence_epoch"].idxmin()]
        insights.append(
            f"⚡ Fastest convergence: {fastest['loss_name']} in {fastest['convergence_epoch']} epochs"
        )

        # Most stable
        most_stable = df.loc[df["stability_score"].idxmax()]
        insights.append(
            f"🎯 Most stable gradients: {most_stable['loss_name']} (score: {most_stable['stability_score']:.3f})"
        )

        # Robustness analysis
        if "robustness_to_noise" in df.columns:
            robust_df = df[df["robustness_to_noise"] > 0]
            if not robust_df.empty:
                most_robust = robust_df.loc[robust_df["robustness_to_noise"].idxmax()]
                insights.append(
                    f"🛡️  Most robust to noise: {most_robust['loss_name']} "
                    f"(retains {most_robust['robustness_to_noise'] * 100:.1f}% performance)"
                )

        # Trade-offs
        time_acc_corr = df["training_time"].corr(df["final_accuracy"])
        if time_acc_corr < -0.5:
            insights.append(
                "⏱️  Trade-off detected: Faster training tends to reduce accuracy"
            )

        return insights

    def plot_comparison(self, save_path: str = None):
        """Generate comparison plots."""
        df = self.create_comparison_table()

        if df.empty:
            print("No data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Loss Function Comparison", fontsize=16, fontweight="bold")

        # Plot 1: Final Accuracy Comparison
        ax1 = axes[0, 0]
        datasets = df["dataset"].unique()
        x = np.arange(len(datasets))
        width = 0.35

        standard_means = []
        novel_means = []

        for dataset in datasets:
            dataset_df = df[df["dataset"] == dataset]
            standard_means.append(
                dataset_df[dataset_df["category"] == "Standard"][
                    "final_accuracy"
                ].mean()
            )
            novel_means.append(
                dataset_df[dataset_df["category"] == "Novel (Ours)"][
                    "final_accuracy"
                ].mean()
            )

        ax1.bar(x - width / 2, standard_means, width, label="Standard", alpha=0.8)
        ax1.bar(x + width / 2, novel_means, width, label="Novel (Ours)", alpha=0.8)
        ax1.set_xlabel("Dataset")
        ax1.set_ylabel("Final Accuracy (%)")
        ax1.set_title("Accuracy Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Training Time
        ax2 = axes[0, 1]
        df.boxplot(column="training_time", by="category", ax=ax2)
        ax2.set_xlabel("Loss Category")
        ax2.set_ylabel("Training Time (s)")
        ax2.set_title("Training Time Distribution")
        plt.suptitle("")  # Remove automatic title

        # Plot 3: Stability Score
        ax3 = axes[1, 0]
        for category in df["category"].unique():
            cat_df = df[df["category"] == category]
            ax3.scatter(
                cat_df["final_accuracy"],
                cat_df["stability_score"],
                label=category,
                alpha=0.7,
                s=100,
            )
        ax3.set_xlabel("Final Accuracy (%)")
        ax3.set_ylabel("Stability Score")
        ax3.set_title("Accuracy vs Stability")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Convergence Speed
        ax4 = axes[1, 1]
        convergence_data = []
        labels = []
        for loss_name in df["loss_name"].unique():
            loss_df = df[df["loss_name"] == loss_name]
            convergence_data.append(loss_df["convergence_epoch"].values)
            labels.append(loss_name.replace(" (Ours)", ""))

        ax4.boxplot(convergence_data, labels=labels)
        ax4.set_xlabel("Loss Function")
        ax4.set_ylabel("Convergence Epoch")
        ax4.set_title("Convergence Speed")
        plt.xticks(rotation=45, ha="right")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"comparison_plot_{timestamp}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")

        plt.close()

    def plot_learning_curves(self, save_path: str = None):
        """Plot learning curves for all loss functions."""
        if not self.results:
            print("No results to plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle("Learning Curves Comparison", fontsize=16, fontweight="bold")

        # Plot 1: Loss curves
        ax1 = axes[0]
        for result in self.results:
            epochs = range(1, len(result["loss_history"]) + 1)
            label = result["loss_name"].replace(" (Ours)", "")
            linestyle = "-" if "Ours" in result["loss_name"] else "--"
            ax1.plot(
                epochs,
                result["loss_history"],
                label=label,
                linestyle=linestyle,
                linewidth=2,
            )

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss Curves")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        # Plot 2: Accuracy curves
        ax2 = axes[1]
        for result in self.results:
            epochs = range(1, len(result["accuracy_history"]) + 1)
            label = result["loss_name"].replace(" (Ours)", "")
            linestyle = "-" if "Ours" in result["loss_name"] else "--"
            ax2.plot(
                epochs,
                result["accuracy_history"],
                label=label,
                linestyle=linestyle,
                linewidth=2,
            )

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training Accuracy Curves")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"learning_curves_{timestamp}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        print(f"Learning curves saved to: {save_path}")
        plt.close()

    def generate_detailed_report(self) -> str:
        """Generate detailed analysis report with insights."""
        df = self.create_comparison_table()
        stats = self.statistical_analysis()
        insights = self.generate_insights()

        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE LOSS FUNCTION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {self.timestamp}")
        report.append(f"Total Loss Functions Tested: {len(df)}")
        report.append(
            f"Datasets: {', '.join(df['dataset'].unique()) if not df.empty else 'N/A'}"
        )

        # Statistical Summary
        report.append("\n" + "-" * 80)
        report.append("STATISTICAL SUMMARY BY CATEGORY")
        report.append("-" * 80)

        for dataset, categories in stats.items():
            report.append(f"\n{dataset.upper()}:")
            for category, metrics in categories.items():
                report.append(f"\n  {category}:")
                report.append(f"    Count: {metrics['count']}")
                report.append(
                    f"    Mean Accuracy: {metrics['mean_accuracy']:.2f}% (±{metrics['std_accuracy']:.2f})"
                )
                report.append(f"    Mean Training Time: {metrics['mean_time']:.2f}s")
                report.append(f"    Mean Stability: {metrics['mean_stability']:.3f}")
                report.append(f"    Best Accuracy: {metrics['best_accuracy']:.2f}%")

        # Key Insights
        report.append("\n" + "-" * 80)
        report.append("KEY INSIGHTS & FINDINGS")
        report.append("-" * 80)

        for insight in insights:
            report.append(f"\n  {insight}")

        # Detailed Comparison
        report.append("\n" + "-" * 80)
        report.append("DETAILED PERFORMANCE COMPARISON")
        report.append("-" * 80)

        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset].sort_values(
                "final_accuracy", ascending=False
            )

            report.append(f"\n{dataset.upper()} Dataset:")
            report.append(
                f"{'Rank':<6}{'Loss Function':<40}{'Accuracy':<12}{'Time':<12}{'Stability':<12}"
            )
            report.append("-" * 80)

            for i, (_, row) in enumerate(dataset_df.iterrows(), 1):
                marker = "✨" if "Ours" in row["loss_name"] else "  "
                report.append(
                    f"{marker}{i:<4}{row['loss_name']:<40}{row['final_accuracy']:<12.2f}"
                    f"{row['training_time']:<12.2f}{row['stability_score']:<12.3f}"
                )

        # Recommendations
        report.append("\n" + "-" * 80)
        report.append("RECOMMENDATIONS")
        report.append("-" * 80)

        # Best for accuracy
        best_acc = df.loc[df["final_accuracy"].idxmax()]
        report.append(f"\n1. BEST FOR ACCURACY:")
        report.append(f"   Use: {best_acc['loss_name']}")
        report.append(f"   Performance: {best_acc['final_accuracy']:.2f}% accuracy")

        # Best for speed
        fastest = df.loc[df["training_time"].idxmin()]
        report.append(f"\n2. BEST FOR SPEED:")
        report.append(f"   Use: {fastest['loss_name']}")
        report.append(f"   Training Time: {fastest['training_time']:.2f}s")

        # Best for stability
        most_stable = df.loc[df["stability_score"].idxmax()]
        report.append(f"\n3. BEST FOR STABILITY:")
        report.append(f"   Use: {most_stable['loss_name']}")
        report.append(f"   Stability Score: {most_stable['stability_score']:.3f}")

        # Novel vs Standard summary
        standard_avg = df[df["category"] == "Standard"]["final_accuracy"].mean()
        novel_avg = df[df["category"] == "Novel (Ours)"]["final_accuracy"].mean()

        report.append(f"\n4. OVERALL COMPARISON:")
        report.append(f"   Standard Losses (avg): {standard_avg:.2f}%")
        report.append(f"   Novel Losses (avg): {novel_avg:.2f}%")
        report.append(
            f"   Improvement: {((novel_avg - standard_avg) / standard_avg * 100):+.1f}%"
        )

        report.append("\n" + "=" * 80)

        report_text = "\n".join(report)

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"detailed_analysis_{timestamp}.txt"
        with open(report_file, "w") as f:
            f.write(report_text)

        print(f"\nDetailed report saved to: {report_file}")
        return report_text


def analyze_benchmark_results(results_file: str = None):
    """
    Analyze benchmark results and generate comprehensive insights.

    Args:
        results_file: Path to benchmark results JSON file
    """
    print("=" * 80)
    print("BENCHMARK ANALYSIS & INSIGHTS GENERATION")
    print("=" * 80)

    analyzer = BenchmarkAnalyzer(results_file)

    # Generate detailed report
    report = analyzer.generate_detailed_report()
    print(report)

    # Generate plots
    print("\nGenerating comparison plots...")
    analyzer.plot_comparison()
    analyzer.plot_learning_curves()

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        # Find most recent results file
        results_dir = Path("loss_framework/experiments/results")
        result_files = sorted(results_dir.glob("benchmark_results_*.json"))
        results_file = result_files[-1] if result_files else None

    analyze_benchmark_results(results_file)
