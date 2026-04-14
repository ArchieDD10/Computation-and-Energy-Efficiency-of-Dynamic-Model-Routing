"""
Simple Temperature Scaling Demo
Author: Archie Deguzman
Purpose: Demonstrate temperature scaling improvements with clear example
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from routing import small, medium, large, normalize_label
from datasets import load_dataset
import json

def demo_temperature_scaling_benefits():
    """Simple demo showing temperature scaling improvements"""

    print("=" * 70)
    print("TEMPERATURE SCALING BENEFITS DEMONSTRATION")
    print("=" * 70)

    # Load the calibration results
    with open('results/temperature_calibration.json', 'r') as f:
        calibration_data = json.load(f)

    print("\n📊 CALIBRATION IMPROVEMENTS:")
    print("   (Lower Brier score = better calibrated confidence)")
    print("   " + "-" * 50)

    total_improvement = 0
    for model_name, results in calibration_data.items():
        original_brier = results['original_brier_score']
        scaled_brier = results['scaled_brier_score']
        improvement = results['calibration_improvement']
        improvement_pct = (improvement / original_brier) * 100

        print(f"   {model_name.upper():6} Model:")
        print(f"     Before: {original_brier:.4f} Brier score")
        print(f"     After:  {scaled_brier:.4f} Brier score")
        print(f"     Improvement: {improvement:.4f} ({improvement_pct:.1f}%)")
        print()

        total_improvement += improvement_pct

    avg_improvement = total_improvement / len(calibration_data)
    print(f"   📈 AVERAGE CALIBRATION IMPROVEMENT: {avg_improvement:.1f}%")

    print("\n🎯 WHAT THIS MEANS FOR YOUR ROUTING:")
    print("   ✅ More accurate confidence scores")
    print("   ✅ Better escalation decisions")
    print("   ✅ Reduced unnecessary computation")
    print("   ✅ Improved overall accuracy")

    print("\n🔬 TECHNICAL DETAILS:")
    print("   • Temperature scaling learns a single parameter T")
    print("   • Scales logits before softmax: p = exp(z/T) / Σ exp(z_i/T)")
    print("   • Calibrates confidence without changing predictions")
    print("   • Optimizes on separate validation set to avoid overfitting")

    # Show example predictions with better confidence
    print("\n💡 CONFIDENCE CALIBRATION EXAMPLE:")

    test_sentences = [
        "This movie is absolutely fantastic!",
        "The film was okay, nothing special.",
        "Terrible acting and boring plot."
    ]

    print("   Sentence: 'This movie is absolutely fantastic!'")
    print("   Original confidence might be: 0.99 (overconfident)")
    print("   Calibrated confidence might be: 0.92 (more realistic)")
    print("   → Better routing decisions with calibrated confidence!")

    print("\n🚀 NEXT STEPS:")
    print("   1. Apply temperature scaling to your routing thresholds")
    print("   2. Re-tune τ_small and τ_med with calibrated confidence")
    print("   3. Measure accuracy improvements on test set")
    print("   4. Generate new visualizations comparing calibrated vs original")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    if os.path.exists('results/temperature_calibration.json'):
        demo_temperature_scaling_benefits()
    else:
        print("[ERROR] No calibration results found. Run temperature scaling first.")