"""
Baseline Runner - Runs single models WITHOUT routing for comparison
Author: Your name here
Purpose: Measure latency of each model tier when used exclusively (no escalation)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from datasets import load_dataset
from routing import small, medium, large, normalize_label
import pandas as pd
import time


def run_single_model(model_pipe, dataset, model_name: str):
    """Run a single model on entire dataset (no routing)"""
    records = []

    for example in dataset:
        text = example["sentence"]
        true_label = example["label"]  # 0 = negative, 1 = positive

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

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("BASELINE RUNNER - Single Model Evaluation")
    print("=" * 60)

    # Load dataset (same as partner's code)
    print("\nLoading SST-2 validation dataset...")
    dataset = load_dataset("glue", "sst2", split="validation")
    print(f"Loaded {len(dataset)} examples")

    # Run each model independently
    print("\n" + "=" * 60)
    print("Running SMALL model (DistilBERT)...")
    print("=" * 60)
    df_small = run_single_model(small, dataset, "small")
    df_small.to_csv("results/baseline_small_only.csv", index=False)
    print(f"[DONE] Completed. Saved to results/baseline_small_only.csv")
    print(f"  Avg latency: {df_small['latency_ms'].mean():.2f}ms")
    print(f"  Accuracy: {(df_small['true_label'] == df_small['pred_label']).mean():.4f}")

    print("\n" + "=" * 60)
    print("Running MEDIUM model (BERT-base)...")
    print("=" * 60)
    df_medium = run_single_model(medium, dataset, "medium")
    df_medium.to_csv("results/baseline_medium_only.csv", index=False)
    print(f"[DONE] Completed. Saved to results/baseline_medium_only.csv")
    print(f"  Avg latency: {df_medium['latency_ms'].mean():.2f}ms")
    print(f"  Accuracy: {(df_medium['true_label'] == df_medium['pred_label']).mean():.4f}")

    print("\n" + "=" * 60)
    print("Running LARGE model (RoBERTa-large)...")
    print("=" * 60)
    df_large = run_single_model(large, dataset, "large")
    df_large.to_csv("results/baseline_large_only.csv", index=False)
    print(f"[DONE] Completed. Saved to results/baseline_large_only.csv")
    print(f"  Avg latency: {df_large['latency_ms'].mean():.2f}ms")
    print(f"  Accuracy: {(df_large['true_label'] == df_large['pred_label']).mean():.4f}")

    print("\n" + "=" * 60)
    print("[DONE] ALL BASELINES COMPLETE")
    print("=" * 60)
    print("\nFiles created:")
    print("  - results/baseline_small_only.csv")
    print("  - results/baseline_medium_only.csv")
    print("  - results/baseline_large_only.csv")


if __name__ == "__main__":
    main()
