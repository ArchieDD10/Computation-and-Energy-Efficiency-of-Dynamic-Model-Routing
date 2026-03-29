"""
Latency Comparison - Compare routing vs baseline approaches
Author: Archie Deguzman 
Purpose: Compare escalation routing against always-small, always-medium, always-large baselines
"""

import pandas as pd
import os


def load_results():
    """Load all result files"""
    results = {}

    # Load routing results (from Kamran's output)
    routing_path = "results/baseline_results.csv"
    if os.path.exists(routing_path):
        results['routing'] = pd.read_csv(routing_path)
    else:
        print(f"[WARNING] Routing results not found: {routing_path}")
        print("   Run: python scripts/run_baseline.py")

    # Load baseline results (my output)
    for model in ['small', 'medium', 'large']:
        path = f"results/baseline_{model}_only.csv"
        if os.path.exists(path):
            results[f'{model}_only'] = pd.read_csv(path)
        else:
            print(f"[WARNING] Baseline not found: {path}")
            print(f"   Run: python analysis/run_baselines.py")

    return results


def compute_metrics(df, approach_name):
    """Compute key metrics for a given approach"""
    accuracy = (df['true_label'] == df['pred_label']).mean()
    avg_latency = df['latency_ms'].mean() if 'latency_ms' in df.columns else df['total_latency_ms'].mean()
    total_latency = df['latency_ms'].sum() if 'latency_ms' in df.columns else df['total_latency_ms'].sum()

    metrics = {
        'approach': approach_name,
        'accuracy': accuracy,
        'avg_latency_ms': avg_latency,
        'total_latency_ms': total_latency,
        'num_samples': len(df),
    }

    # For routing approach, adding model usage stats
    if 'chosen_model' in df.columns:
        model_counts = df['chosen_model'].value_counts()
        metrics['small_usage'] = model_counts.get('small', 0)
        metrics['medium_usage'] = model_counts.get('medium', 0)
        metrics['large_usage'] = model_counts.get('large', 0)
        metrics['small_pct'] = (model_counts.get('small', 0) / len(df)) * 100
        metrics['medium_pct'] = (model_counts.get('medium', 0) / len(df)) * 100
        metrics['large_pct'] = (model_counts.get('large', 0) / len(df)) * 100

        if 'num_models_used' in df.columns:
            metrics['avg_models_per_query'] = df['num_models_used'].mean()

    return metrics


def print_comparison_table(metrics_list):
    """Pretty print comparison table"""
    print("\n" + "=" * 100)
    print("LATENCY & ACCURACY COMPARISON")
    print("=" * 100)

    # Header
    print(f"{'Approach':<20} {'Accuracy':>10} {'Avg Latency':>15} {'Total Latency':>15} {'Speedup':>12}")
    print("-" * 100)

    # Find baseline (large_only) for speedup calculation
    baseline_latency = None
    for m in metrics_list:
        if m['approach'] == 'Always Large':
            baseline_latency = m['avg_latency_ms']
            break

    # Print each approach
    for m in metrics_list:
        speedup = baseline_latency / m['avg_latency_ms'] if baseline_latency else 1.0
        print(f"{m['approach']:<20} {m['accuracy']:>10.4f} {m['avg_latency_ms']:>12.2f}ms "
              f"{m['total_latency_ms']:>12.1f}ms {speedup:>11.2f}x")

    print("=" * 100)


def print_routing_details(metrics):
    """Print detailed routing statistics"""
    if 'small_usage' in metrics:
        print("\n" + "=" * 60)
        print("ROUTING MODEL USAGE (Escalation Strategy)")
        print("=" * 60)
        print(f"Small model:  {metrics['small_usage']:>4} queries ({metrics['small_pct']:>5.1f}%)")
        print(f"Medium model: {metrics['medium_usage']:>4} queries ({metrics['medium_pct']:>5.1f}%)")
        print(f"Large model:  {metrics['large_usage']:>4} queries ({metrics['large_pct']:>5.1f}%)")
        print(f"\nAverage models invoked per query: {metrics.get('avg_models_per_query', 'N/A'):.2f}")
        print("=" * 60)


def compute_latency_savings(metrics_list):
    """Compute latency savings compared to always-large"""
    print("\n" + "=" * 60)
    print("LATENCY SAVINGS vs Always Large Baseline")
    print("=" * 60)

    # Find large baseline
    large_metrics = next((m for m in metrics_list if m['approach'] == 'Always Large'), None)
    if not large_metrics:
        print("[WARNING] Large baseline not found")
        return

    baseline_total = large_metrics['total_latency_ms']

    for m in metrics_list:
        if m['approach'] == 'Always Large':
            continue

        savings_ms = baseline_total - m['total_latency_ms']
        savings_pct = (savings_ms / baseline_total) * 100

        print(f"{m['approach']:<20} saved {savings_ms:>10.1f}ms ({savings_pct:>5.1f}%)")

    print("=" * 60)


def main():
    print("=" * 60)
    print("LATENCY COMPARISON ANALYSIS")
    print("=" * 60)

    # Load all results
    results = load_results()

    if len(results) == 0:
        print("\n[ERROR] No results found. Please run:")
        print("   1. python scripts/run_baseline.py (partner's routing code)")
        print("   2. python analysis/run_baselines.py (your baseline code)")
        return

    # Compute metrics for each approach
    metrics_list = []

    if 'routing' in results:
        metrics_list.append(compute_metrics(results['routing'], 'Routing (Escalation)'))

    if 'small_only' in results:
        metrics_list.append(compute_metrics(results['small_only'], 'Always Small'))

    if 'medium_only' in results:
        metrics_list.append(compute_metrics(results['medium_only'], 'Always Medium'))

    if 'large_only' in results:
        metrics_list.append(compute_metrics(results['large_only'], 'Always Large'))

    # Print comparison
    print_comparison_table(metrics_list)

    # Print routing details if available
    routing_metrics = next((m for m in metrics_list if m['approach'] == 'Routing (Escalation)'), None)
    if routing_metrics:
        print_routing_details(routing_metrics)

    # Compute savings
    compute_latency_savings(metrics_list)

    # Save summary
    summary_df = pd.DataFrame(metrics_list)
    summary_df.to_csv("results/comparison_summary.csv", index=False)
    print("\n[DONE] Summary saved to: results/comparison_summary.csv")


if __name__ == "__main__":
    main()
