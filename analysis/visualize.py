"""
Visualization Module - Generate graphs for latency and accuracy analysis
Author: Archie Deguzman
Purpose: Create publication-quality graphs comparing routing vs baseline approaches
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_results():
    """Load all result files"""
    results = {}

    files = {
        'routing': 'results/baseline_results.csv',
        'small_only': 'results/baseline_small_only.csv',
        'medium_only': 'results/baseline_medium_only.csv',
        'large_only': 'results/baseline_large_only.csv',
    }

    for key, path in files.items():
        if os.path.exists(path):
            results[key] = pd.read_csv(path)
        else:
            print(f"[WARNING] File not found: {path}")

    return results


def plot_latency_comparison(results, save_path="results/figures/latency_comparison.png"):
    """Bar chart comparing average latency across approaches"""
    fig, ax = plt.subplots(figsize=(10, 6))

    approaches = []
    latencies = []
    colors = []

    if 'small_only' in results:
        approaches.append('Always\nSmall')
        latencies.append(results['small_only']['latency_ms'].mean())
        colors.append('#3498db')  # Blue

    if 'medium_only' in results:
        approaches.append('Always\nMedium')
        latencies.append(results['medium_only']['latency_ms'].mean())
        colors.append('#f39c12')  # Orange

    if 'large_only' in results:
        approaches.append('Always\nLarge')
        latencies.append(results['large_only']['latency_ms'].mean())
        colors.append('#e74c3c')  # Red

    if 'routing' in results:
        approaches.append('Routing\n(Escalation)')
        latency_col = 'total_latency_ms' if 'total_latency_ms' in results['routing'].columns else 'latency_ms'
        latencies.append(results['routing'][latency_col].mean())
        colors.append('#2ecc71')  # Green

    bars = ax.bar(approaches, latencies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, latency in zip(bars, latencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{latency:.1f}ms',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Average Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Average Inference Latency Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved: {save_path}")
    plt.close()


def plot_accuracy_vs_latency(results, save_path="results/figures/accuracy_vs_latency.png"):
    """Scatter plot showing accuracy vs latency tradeoff"""
    fig, ax = plt.subplots(figsize=(10, 7))

    data = []

    for key in ['small_only', 'medium_only', 'large_only', 'routing']:
        if key not in results:
            continue

        df = results[key]
        accuracy = (df['true_label'] == df['pred_label']).mean()

        if key == 'routing':
            latency = df['total_latency_ms'].mean() if 'total_latency_ms' in df.columns else df['latency_ms'].mean()
            label = 'Routing (Escalation)'
            color = '#2ecc71'
            marker = 's'
            size = 300
        elif key == 'small_only':
            latency = df['latency_ms'].mean()
            label = 'Always Small'
            color = '#3498db'
            marker = 'o'
            size = 250
        elif key == 'medium_only':
            latency = df['latency_ms'].mean()
            label = 'Always Medium'
            color = '#f39c12'
            marker = 'o'
            size = 250
        else:  # large_only
            latency = df['latency_ms'].mean()
            label = 'Always Large'
            color = '#e74c3c'
            marker = 'o'
            size = 250

        ax.scatter(latency, accuracy * 100, s=size, c=color, marker=marker,
                   alpha=0.8, edgecolors='black', linewidths=2, label=label)

        # Add annotation
        ax.annotate(f'{accuracy*100:.2f}%\n{latency:.1f}ms',
                    (latency, accuracy * 100),
                    textcoords="offset points",
                    xytext=(0, 15),
                    ha='center',
                    fontsize=10,
                    fontweight='bold')

    ax.set_xlabel('Average Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Accuracy vs Latency Tradeoff', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved: {save_path}")
    plt.close()


def plot_model_usage_distribution(results, save_path="results/figures/model_usage.png"):
    """Pie chart showing model usage in routing approach"""
    if 'routing' not in results:
        print("[WARNING] Routing results not found, skipping model usage plot")
        return

    df = results['routing']
    if 'chosen_model' not in df.columns:
        print("[WARNING] No 'chosen_model' column found")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    model_counts = df['chosen_model'].value_counts()
    colors = {'small': '#3498db', 'medium': '#f39c12', 'large': '#e74c3c'}

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        model_counts.values,
        labels=[f'{m.title()}\n({model_counts[m]} queries)' for m in model_counts.index],
        colors=[colors.get(m, '#95a5a6') for m in model_counts.index],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        explode=[0.05] * len(model_counts)
    )

    # Style percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(13)
        autotext.set_fontweight('bold')

    ax.set_title('Model Usage Distribution in Routing Strategy',
                 fontsize=15, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved: {save_path}")
    plt.close()


def plot_latency_distribution(results, save_path="results/figures/latency_distribution.png"):
    """Violin plot comparing latency distributions"""
    fig, ax = plt.subplots(figsize=(12, 7))

    data_to_plot = []
    labels = []
    colors = []

    if 'small_only' in results:
        data_to_plot.append(results['small_only']['latency_ms'])
        labels.append('Always\nSmall')
        colors.append('#3498db')

    if 'medium_only' in results:
        data_to_plot.append(results['medium_only']['latency_ms'])
        labels.append('Always\nMedium')
        colors.append('#f39c12')

    if 'large_only' in results:
        data_to_plot.append(results['large_only']['latency_ms'])
        labels.append('Always\nLarge')
        colors.append('#e74c3c')

    if 'routing' in results:
        latency_col = 'total_latency_ms' if 'total_latency_ms' in results['routing'].columns else 'latency_ms'
        data_to_plot.append(results['routing'][latency_col])
        labels.append('Routing\n(Escalation)')
        colors.append('#2ecc71')

    parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)),
                          showmeans=True, showmedians=True)

    # Color the violin plots
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Latency Distribution Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved: {save_path}")
    plt.close()


def plot_cumulative_latency(results, save_path="results/figures/cumulative_latency.png"):
    """Line plot showing cumulative latency over samples"""
    fig, ax = plt.subplots(figsize=(12, 7))

    for key, label, color in [
        ('small_only', 'Always Small', '#3498db'),
        ('medium_only', 'Always Medium', '#f39c12'),
        ('large_only', 'Always Large', '#e74c3c'),
        ('routing', 'Routing (Escalation)', '#2ecc71'),
    ]:
        if key not in results:
            continue

        df = results[key]
        latency_col = 'latency_ms' if key != 'routing' else (
            'total_latency_ms' if 'total_latency_ms' in df.columns else 'latency_ms'
        )

        cumulative = df[latency_col].cumsum()
        ax.plot(range(len(cumulative)), cumulative, label=label, color=color, linewidth=2.5)

    ax.set_xlabel('Number of Queries', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Cumulative Latency Over Time', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved: {save_path}")
    plt.close()


def plot_speedup_comparison(results, save_path="results/figures/speedup_comparison.png"):
    """Bar chart showing speedup relative to always-large baseline"""
    if 'large_only' not in results:
        print("[WARNING] Large baseline not found, skipping speedup plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_latency = results['large_only']['latency_ms'].mean()

    approaches = []
    speedups = []
    colors = []

    if 'small_only' in results:
        approaches.append('Always\nSmall')
        speedups.append(baseline_latency / results['small_only']['latency_ms'].mean())
        colors.append('#3498db')

    if 'medium_only' in results:
        approaches.append('Always\nMedium')
        speedups.append(baseline_latency / results['medium_only']['latency_ms'].mean())
        colors.append('#f39c12')

    if 'routing' in results:
        approaches.append('Routing\n(Escalation)')
        latency_col = 'total_latency_ms' if 'total_latency_ms' in results['routing'].columns else 'latency_ms'
        speedups.append(baseline_latency / results['routing'][latency_col].mean())
        colors.append('#2ecc71')

    # Add baseline reference
    approaches.append('Always\nLarge')
    speedups.append(1.0)
    colors.append('#e74c3c')

    bars = ax.bar(approaches, speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add baseline reference line
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline (Always Large)')

    ax.set_ylabel('Speedup vs Always Large', fontsize=13, fontweight='bold')
    ax.set_title('Speedup Comparison (Relative to Always Large)', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved: {save_path}")
    plt.close()


def generate_all_plots(results):
    """Generate all visualization plots"""
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60 + "\n")

    plot_latency_comparison(results)
    plot_accuracy_vs_latency(results)
    plot_model_usage_distribution(results)
    plot_latency_distribution(results)
    plot_cumulative_latency(results)
    plot_speedup_comparison(results)

    print("\n" + "=" * 60)
    print("[DONE] ALL PLOTS GENERATED")
    print("=" * 60)
    print("\nPlots saved to: results/figures/")


def main():
    print("=" * 60)
    print("VISUALIZATION MODULE")
    print("=" * 60)

    results = load_results()

    if len(results) == 0:
        print("\n[ERROR] No results found. Please run:")
        print("   1. python scripts/run_baseline.py")
        print("   2. python analysis/run_baselines.py")
        return

    generate_all_plots(results)


if __name__ == "__main__":
    main()
