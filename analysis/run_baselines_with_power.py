"""
Baseline Runner with Power Monitoring
Author: Archie Deguzman
Purpose: Measure latency AND power consumption of each model tier
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from datasets import load_dataset
from routing import small, medium, large, normalize_label
import pandas as pd
import time

# Import power monitor
from power_monitor import PowerMonitor


def run_single_model_with_power(model_pipe, dataset, model_name: str):
    """Run a single model on entire dataset with power monitoring"""
    records = []
    power_monitor = PowerMonitor()

    # Start power monitoring
    if power_monitor.gpu_handle:
        print(f"  [INFO] GPU power monitoring active")
        power_monitor.start(interval_ms=10)
    else:
        print(f"  [WARNING] GPU power monitoring not available")

    for example in dataset:
        text = example["sentence"]
        true_label = example["label"]

        # Time the inference
        t0 = time.perf_counter()
        scores = model_pipe(text)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Get prediction
        best = max(scores, key=lambda d: d["score"])
        pred_label_str = normalize_label(best["label"])
        pred_label = 1 if pred_label_str == "POSITIVE" else 0
        confidence = float(best["score"])

        records.append({
            "text": text,
            "true_label": true_label,
            "pred_label": pred_label,
            "model": model_name,
            "confidence": confidence,
            "latency_ms": latency_ms,
        })

    # Stop power monitoring and get stats
    power_stats = power_monitor.stop()

    df = pd.DataFrame(records)

    # Add power statistics as metadata
    return df, power_stats


def main():
    print("=" * 60)
    print("BASELINE RUNNER - With Power Monitoring")
    print("=" * 60)

    # Load dataset
    print("\nLoading SST-2 validation dataset...")
    dataset = load_dataset("glue", "sst2", split="validation")
    print(f"Loaded {len(dataset)} examples")

    all_power_stats = {}

    # Run each model independently
    print("\n" + "=" * 60)
    print("Running SMALL model (DistilBERT)...")
    print("=" * 60)
    df_small, power_small = run_single_model_with_power(small, dataset, "small")
    df_small.to_csv("results/baseline_small_only.csv", index=False)
    all_power_stats['small'] = power_small
    print(f"[DONE] Completed. Saved to results/baseline_small_only.csv")
    print(f"  Avg latency: {df_small['latency_ms'].mean():.2f}ms")
    print(f"  Accuracy: {(df_small['true_label'] == df_small['pred_label']).mean():.4f}")
    if power_small['num_samples'] > 0:
        print(f"  Avg power: {power_small['avg_power_watts']:.2f}W")
        print(f"  Total energy: {power_small['energy_joules']:.2f}J ({power_small['energy_wh']:.4f}Wh)")

    print("\n" + "=" * 60)
    print("Running MEDIUM model (BERT-base)...")
    print("=" * 60)
    df_medium, power_medium = run_single_model_with_power(medium, dataset, "medium")
    df_medium.to_csv("results/baseline_medium_only.csv", index=False)
    all_power_stats['medium'] = power_medium
    print(f"[DONE] Completed. Saved to results/baseline_medium_only.csv")
    print(f"  Avg latency: {df_medium['latency_ms'].mean():.2f}ms")
    print(f"  Accuracy: {(df_medium['true_label'] == df_medium['pred_label']).mean():.4f}")
    if power_medium['num_samples'] > 0:
        print(f"  Avg power: {power_medium['avg_power_watts']:.2f}W")
        print(f"  Total energy: {power_medium['energy_joules']:.2f}J ({power_medium['energy_wh']:.4f}Wh)")

    print("\n" + "=" * 60)
    print("Running LARGE model (RoBERTa-large)...")
    print("=" * 60)
    df_large, power_large = run_single_model_with_power(large, dataset, "large")
    df_large.to_csv("results/baseline_large_only.csv", index=False)
    all_power_stats['large'] = power_large
    print(f"[DONE] Completed. Saved to results/baseline_large_only.csv")
    print(f"  Avg latency: {df_large['latency_ms'].mean():.2f}ms")
    print(f"  Accuracy: {(df_large['true_label'] == df_large['pred_label']).mean():.4f}")
    if power_large['num_samples'] > 0:
        print(f"  Avg power: {power_large['avg_power_watts']:.2f}W")
        print(f"  Total energy: {power_large['energy_joules']:.2f}J ({power_large['energy_wh']:.4f}Wh)")

    print("\n" + "=" * 60)
    print("[DONE] ALL BASELINES COMPLETE")
    print("=" * 60)
    print("\nFiles created:")
    print("  - results/baseline_small_only.csv")
    print("  - results/baseline_medium_only.csv")
    print("  - results/baseline_large_only.csv")

    # Save power statistics
    import json
    with open("results/power_stats_baselines.json", "w") as f:
        json.dump(all_power_stats, f, indent=2)
    print("  - results/power_stats_baselines.json")

    # Print power comparison
    if all(stats['num_samples'] > 0 for stats in all_power_stats.values()):
        print("\n" + "=" * 60)
        print("POWER CONSUMPTION COMPARISON")
        print("=" * 60)
        for model, stats in all_power_stats.items():
            print(f"{model.upper():8} | Avg Power: {stats['avg_power_watts']:6.2f}W | "
                  f"Total Energy: {stats['energy_joules']:8.2f}J ({stats['energy_wh']:.4f}Wh)")
        print("=" * 60)


if __name__ == "__main__":
    main()
