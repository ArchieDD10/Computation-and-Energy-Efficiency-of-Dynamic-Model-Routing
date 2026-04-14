"""
Enhanced Baseline Runner - Latency + Power Monitoring
Author: Archie Deguzman
Purpose: Measure both latency and power consumption for model comparison
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from datasets import load_dataset
from routing import small, medium, large, normalize_label
import pandas as pd
import time
import json

# Import our working power monitor
from simple_power_monitor import SimpleInferenceProfiler, profile_model_inference


def run_single_model_with_power(model_pipe, dataset, model_name: str):
    """Run a single model on entire dataset with power monitoring"""
    print(f"  [INFO] Running {model_name} model with power monitoring...")

    # Take a sample of texts for power profiling
    sample_texts = [example["sentence"] for example in dataset.select(range(min(20, len(dataset))))]

    # Profile power consumption on sample
    power_profile = profile_model_inference(model_pipe, sample_texts, model_name)

    print(f"  [INFO] Power profile: {power_profile['avg_power_watts']:.1f}W avg, "
          f"{power_profile['avg_latency_ms']:.1f}ms avg")

    # Now run full dataset for accuracy measurement
    records = []
    total_latency = 0
    profiler = SimpleInferenceProfiler()
    profiler.start_profiling()

    for i, example in enumerate(dataset):
        text = example["sentence"]
        true_label = example["label"]

        # Time the inference
        t0 = time.perf_counter()
        scores = model_pipe(text)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        total_latency += latency_ms

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

        # Progress update
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{len(dataset)} samples")

    # Get overall power stats for full run
    full_power_stats = profiler.end_profiling()

    df = pd.DataFrame(records)

    # Combine power and performance stats
    combined_stats = {
        'model': model_name,
        'samples': len(dataset),
        'avg_latency_ms': df['latency_ms'].mean(),
        'total_latency_ms': total_latency,
        'accuracy': (df['true_label'] == df['pred_label']).mean(),
        'power_method': full_power_stats.get('power_method', 'unknown'),
        'avg_power_watts': full_power_stats.get('avg_power_watts', power_profile.get('avg_power_watts', 0)),
        'total_energy_joules': full_power_stats.get('energy_joules', 0),
        'total_energy_wh': full_power_stats.get('energy_wh', 0),
        'confidence_level': full_power_stats.get('confidence', 'medium')
    }

    return df, combined_stats


def main():
    print("=" * 70)
    print("ENHANCED BASELINE RUNNER - Latency + Power Analysis")
    print("=" * 70)

    # Load dataset
    print("\nLoading SST-2 validation dataset...")
    dataset = load_dataset("glue", "sst2", split="validation")
    print(f"Loaded {len(dataset)} examples")

    all_stats = {}

    # Run each model independently with power monitoring
    models = [
        (small, "small", "DistilBERT"),
        (medium, "medium", "BERT-base"),
        (large, "large", "RoBERTa-large")
    ]

    for model_pipe, model_key, model_desc in models:
        print(f"\n" + "=" * 70)
        print(f"Running {model_desc.upper()} model...")
        print("=" * 70)

        df, stats = run_single_model_with_power(model_pipe, dataset, model_key)

        # Save CSV
        csv_path = f"results/baseline_{model_key}_only.csv"
        df.to_csv(csv_path, index=False)

        # Store stats
        all_stats[model_key] = stats

        # Print summary
        print(f"[DONE] Completed {model_desc}")
        print(f"  File: {csv_path}")
        print(f"  Avg latency: {stats['avg_latency_ms']:.2f}ms")
        print(f"  Accuracy: {stats['accuracy']:.4f}")
        print(f"  Avg power: {stats['avg_power_watts']:.1f}W")
        print(f"  Total energy: {stats['total_energy_joules']:.1f}J ({stats['total_energy_wh']:.4f}Wh)")
        print(f"  Power confidence: {stats['confidence_level']}")

    # Save detailed power statistics
    with open("results/power_analysis.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n" + "=" * 70)
    print("[DONE] ALL ENHANCED BASELINES COMPLETE")
    print("=" * 70)
    print("\nFiles created:")
    print("  - results/baseline_small_only.csv")
    print("  - results/baseline_medium_only.csv")
    print("  - results/baseline_large_only.csv")
    print("  - results/power_analysis.json")

    # Print power efficiency comparison
    print(f"\n" + "=" * 70)
    print("POWER EFFICIENCY COMPARISON")
    print("=" * 70)

    for model_key, stats in all_stats.items():
        power = stats['avg_power_watts']
        latency = stats['avg_latency_ms']
        accuracy = stats['accuracy']
        energy_per_query = (power * latency / 1000)  # Joules per query

        print(f"{model_key.upper():8} | "
              f"Power: {power:5.1f}W | "
              f"Latency: {latency:6.1f}ms | "
              f"Accuracy: {accuracy:.3f} | "
              f"Energy/query: {energy_per_query:.3f}J")

    # Calculate efficiency metrics
    print(f"\n" + "=" * 70)
    print("EFFICIENCY METRICS (Energy per Correct Prediction)")
    print("=" * 70)

    for model_key, stats in all_stats.items():
        power = stats['avg_power_watts']
        latency = stats['avg_latency_ms']
        accuracy = stats['accuracy']

        energy_per_query = (power * latency / 1000)  # Joules per query
        energy_per_correct = energy_per_query / accuracy if accuracy > 0 else float('inf')

        print(f"{model_key.upper():8} | Energy per correct prediction: {energy_per_correct:.3f}J")

    print("=" * 70)


if __name__ == "__main__":
    main()
