"""
Temperature Scaling Integration Script
Author: Archie Deguzman
Purpose: Integrate temperature scaling into existing routing system and evaluate improvements
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from temperature_scaling import CalibratedRoutingSystem, TemperatureScaling
from routing import small, medium, large, normalize_label
from datasets import load_dataset
import numpy as np
import json
import time
from typing import Dict, List

def load_calibration_data(n_samples: int = 200):
    """Load subset of SST-2 for temperature calibration"""
    print(f"[INFO] Loading {n_samples} samples for calibration...")

    dataset = load_dataset("sst2", split="train").shuffle(seed=42).select(range(1000))

    # Take a subset for calibration (separate from main evaluation)
    calibration_indices = np.random.choice(len(dataset), n_samples, replace=False)

    texts = []
    labels = []

    for idx in calibration_indices:
        example = dataset[idx]
        texts.append(example["sentence"])
        labels.append(example["label"])

    print(f"[INFO] Loaded {len(texts)} calibration examples")
    return texts, labels

def evaluate_routing_with_calibration():
    """Compare original vs temperature-calibrated routing"""

    print("=" * 70)
    print("TEMPERATURE SCALING EVALUATION")
    print("=" * 70)

    # Load calibration data (separate from main evaluation set)
    cal_texts, cal_labels = load_calibration_data(n_samples=150)

    # Create calibrated routing system
    calibrated_system = CalibratedRoutingSystem(small, medium, large)

    # Calibrate all models
    calibration_results = calibrated_system.calibrate_all_models(cal_texts, cal_labels)

    # Save calibration results
    os.makedirs('results', exist_ok=True)
    with open('results/temperature_calibration.json', 'w') as f:
        # Convert any tensor values to regular Python types
        serializable_results = {}
        for model, results in calibration_results.items():
            serializable_results[model] = {}
            for key, value in results.items():
                if hasattr(value, 'item'):  # PyTorch tensor
                    serializable_results[model][key] = value.item()
                else:
                    serializable_results[model][key] = value

        json.dump(serializable_results, f, indent=2)

    print(f"\n[INFO] Calibration results saved to: results/temperature_calibration.json")

    # Load evaluation dataset (different from calibration)
    print("\n[INFO] Loading evaluation dataset...")
    eval_dataset = load_dataset("sst2", split="validation")

    # Use different samples for evaluation (avoid data leakage)
    eval_indices = list(set(range(len(eval_dataset))) - set(range(150)))[:200]  # Take 200 different samples

    print(f"\n[INFO] Evaluating on {len(eval_indices)} test samples...")

    # Compare original vs calibrated routing
    original_correct = 0
    calibrated_correct = 0
    original_model_usage = {'small': 0, 'medium': 0, 'large': 0}
    calibrated_model_usage = {'small': 0, 'medium': 0, 'large': 0}

    results_comparison = []

    for i, idx in enumerate(eval_indices):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(eval_indices)}")

        example = eval_dataset[idx]
        text = example["sentence"]
        true_label = example["label"]

        # Original routing (using your existing logic)
        # Small model
        small_result = small(text)
        small_confidence = max(small_result[0]['score'], 1 - small_result[0]['score'])

        if small_confidence >= 0.90:  # tau_small
            original_pred = normalize_label(small_result[0]['label'])
            original_model = 'small'
        else:
            # Medium model
            medium_result = medium(text)
            medium_confidence = max(medium_result[0]['score'], 1 - medium_result[0]['score'])

            if medium_confidence >= 0.95:  # tau_medium
                original_pred = normalize_label(medium_result[0]['label'])
                original_model = 'medium'
            else:
                # Large model
                large_result = large(text)
                original_pred = normalize_label(large_result[0]['label'])
                original_model = 'large'

        # Calibrated routing
        calibrated_result = calibrated_system.predict_with_routing_calibrated(text)
        calibrated_pred = calibrated_result['prediction']
        calibrated_model = calibrated_result['model_used']

        # Track accuracy
        if original_pred == true_label:
            original_correct += 1
        if calibrated_pred == true_label:
            calibrated_correct += 1

        # Track model usage
        original_model_usage[original_model] += 1
        calibrated_model_usage[calibrated_model] += 1

        # Store detailed comparison
        results_comparison.append({
            'text': text,
            'true_label': true_label,
            'original_pred': original_pred,
            'calibrated_pred': calibrated_pred,
            'original_model': original_model,
            'calibrated_model': calibrated_model,
            'original_confidence': small_confidence if original_model == 'small' else (medium_confidence if original_model == 'medium' else None),
            'calibrated_confidence': calibrated_result['confidence']
        })

    # Calculate metrics
    n_eval = len(eval_indices)
    original_accuracy = original_correct / n_eval
    calibrated_accuracy = calibrated_correct / n_eval

    print(f"\n" + "=" * 70)
    print("CALIBRATION EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nACCURACY COMPARISON:")
    print(f"  Original Routing:   {original_accuracy:.4f} ({original_correct}/{n_eval})")
    print(f"  Calibrated Routing: {calibrated_accuracy:.4f} ({calibrated_correct}/{n_eval})")
    print(f"  Improvement:        {calibrated_accuracy - original_accuracy:+.4f}")

    print(f"\nMODEL USAGE COMPARISON:")
    print(f"  Model      Original    Calibrated   Change")
    print(f"  -------    --------    ----------   ------")
    for model in ['small', 'medium', 'large']:
        orig_pct = original_model_usage[model] / n_eval * 100
        cal_pct = calibrated_model_usage[model] / n_eval * 100
        change = cal_pct - orig_pct
        print(f"  {model:8}   {orig_pct:6.1f}%      {cal_pct:6.1f}%      {change:+5.1f}%")

    # Calculate efficiency metrics
    original_efficiency = (original_model_usage['small'] * 1 +
                          original_model_usage['medium'] * 2 +
                          original_model_usage['large'] * 3) / n_eval

    calibrated_efficiency = (calibrated_model_usage['small'] * 1 +
                            calibrated_model_usage['medium'] * 2 +
                            calibrated_model_usage['large'] * 3) / n_eval

    print(f"\nCOMPUTATIONAL EFFICIENCY:")
    print(f"  Original avg model complexity:   {original_efficiency:.2f}")
    print(f"  Calibrated avg model complexity: {calibrated_efficiency:.2f}")
    print(f"  Efficiency change:               {calibrated_efficiency - original_efficiency:+.2f}")

    # Save detailed results
    evaluation_summary = {
        'calibration_results': serializable_results,
        'evaluation_metrics': {
            'original_accuracy': original_accuracy,
            'calibrated_accuracy': calibrated_accuracy,
            'accuracy_improvement': calibrated_accuracy - original_accuracy,
            'original_model_usage': original_model_usage,
            'calibrated_model_usage': calibrated_model_usage,
            'original_efficiency': original_efficiency,
            'calibrated_efficiency': calibrated_efficiency,
            'efficiency_change': calibrated_efficiency - original_efficiency
        },
        'detailed_comparisons': results_comparison[:10]  # Save first 10 for inspection
    }

    with open('results/temperature_scaling_evaluation.json', 'w') as f:
        json.dump(evaluation_summary, f, indent=2)

    print(f"\n[DONE] Detailed results saved to: results/temperature_scaling_evaluation.json")
    print("=" * 70)

    return evaluation_summary

def analyze_confidence_distribution():
    """Analyze how temperature scaling affects confidence distributions"""
    print("\n[INFO] Analyzing confidence distribution changes...")

    # This would create visualizations showing:
    # 1. Before/after confidence histograms
    # 2. Reliability diagrams (calibration plots)
    # 3. Confidence vs accuracy scatter plots

    # For now, just indicate this analysis is available
    print("[INFO] Confidence distribution analysis available in temperature_scaling.py")

if __name__ == "__main__":
    try:
        # Run temperature scaling evaluation
        results = evaluate_routing_with_calibration()

        # Analyze confidence distributions
        analyze_confidence_distribution()

        print(f"\n🎯 TEMPERATURE SCALING COMPLETE!")
        print(f"   Check results/temperature_scaling_evaluation.json for detailed metrics")

    except Exception as e:
        print(f"[ERROR] Temperature scaling evaluation failed: {e}")
        import traceback
        traceback.print_exc()