"""
Master Script - Run complete analysis pipeline
Author: Archie Deguzman 
Purpose: Run all baseline comparisons and generate visualizations
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "=" * 70)
    print(f">> {description}")
    print("=" * 70)

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Running: {cmd}")
        print(f"Error message: {e.stderr}")
        return False


def main():
    print("=" * 70)
    print("COMPLETE ANALYSIS PIPELINE")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Run single-model baselines with power monitoring (small, medium, large)")
    print("  2. Compare routing vs baselines")
    print("  3. Generate all visualizations")
    print("\nNote: Your partner's routing results should already exist at:")
    print("      results/baseline_results.csv")
    print("\n" + "=" * 70)

    # Check if routing results exist
    if not os.path.exists("results/baseline_results.csv"):
        print("\n[WARNING] Routing results not found!")
        print("   Your partner needs to run: python scripts/run_baseline.py")
        print("   Or you can run it yourself if the code is ready.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return

    # Step 1: Run baselines WITH power monitoring
    success = run_command(
        f'python analysis/run_enhanced_baselines.py',
        "Step 1/3: Running single-model baselines with power monitoring"
    )
    if not success:
        print("\n[ERROR] Pipeline stopped due to error in baseline generation")
        return

    # Step 2: Compare results
    success = run_command(
        f'python analysis/compare_latency.py',
        "Step 2/3: Comparing routing vs baselines"
    )
    if not success:
        print("\n[WARNING] Comparison failed, but continuing to visualization...")

    # Step 3: Generate visualizations
    success = run_command(
        f'python analysis/visualize.py',
        "Step 3/3: Generating visualizations"
    )

    # Summary
    print("\n" + "=" * 70)
    print("[DONE] PIPELINE COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  [DATA] Baselines:")
    print("     - results/baseline_small_only.csv")
    print("     - results/baseline_medium_only.csv")
    print("     - results/baseline_large_only.csv")
    print("\n  [ANALYSIS] Summary:")
    print("     - results/comparison_summary.csv")
    print("     - results/power_analysis.json")
    print("\n  [GRAPHS] Visualizations:")
    print("     - results/figures/latency_comparison.png")
    print("     - results/figures/accuracy_vs_latency.png")
    print("     - results/figures/model_usage.png")
    print("     - results/figures/latency_distribution.png")
    print("     - results/figures/cumulative_latency.png")
    print("     - results/figures/speedup_comparison.png")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
