"""
Power Consumption Visualization Module
Author: Archie Deguzman
Purpose: Create power-focused graphs showing energy efficiency of routing vs baselines
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_power_data():
    """Load power analysis data"""
    with open('results/power_analysis.json', 'r') as f:
        power_data = json.load(f)
    return power_data

def plot_power_comparison(save_path="results/figures/power_comparison.png"):
    """Bar chart comparing average power consumption"""
    power_data = load_power_data()

    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['Small', 'Medium', 'Large']
    powers = [power_data['small']['avg_power_watts'],
              power_data['medium']['avg_power_watts'],
              power_data['large']['avg_power_watts']]
    colors = ['#3498db', '#f39c12', '#e74c3c']

    bars = ax.bar(models, powers, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, power in zip(bars, powers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{power:.1f}W',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Average Power Consumption (Watts)', fontsize=13, fontweight='bold')
    ax.set_title('Average Power Consumption by Model Size', fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved: {save_path}")
    plt.close()

def plot_energy_efficiency(save_path="results/figures/energy_efficiency.png"):
    """Bar chart comparing energy per query and per correct prediction"""
    power_data = load_power_data()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    models = ['Small', 'Medium', 'Large']
    colors = ['#3498db', '#f39c12', '#e74c3c']

    # Energy per query
    energy_per_query = []
    for model in ['small', 'medium', 'large']:
        total_energy = power_data[model]['total_energy_joules']
        samples = power_data[model]['samples']
        energy_per_query.append(total_energy / samples)

    bars1 = ax1.bar(models, energy_per_query, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, energy in zip(bars1, energy_per_query):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.3f}J',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Energy per Query (Joules)', fontsize=12, fontweight='bold')
    ax1.set_title('Energy Consumption per Query', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Energy per correct prediction
    energy_per_correct = []
    for model in ['small', 'medium', 'large']:
        total_energy = power_data[model]['total_energy_joules']
        samples = power_data[model]['samples']
        accuracy = power_data[model]['accuracy']
        energy_per_correct.append(total_energy / (samples * accuracy))

    bars2 = ax2.bar(models, energy_per_correct, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, energy in zip(bars2, energy_per_correct):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.3f}J',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_ylabel('Energy per Correct Prediction (Joules)', fontsize=12, fontweight='bold')
    ax2.set_title('Energy Efficiency (per Correct Answer)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved: {save_path}")
    plt.close()

def plot_power_vs_latency(save_path="results/figures/power_vs_latency.png"):
    """Scatter plot showing power vs latency tradeoff"""
    power_data = load_power_data()

    fig, ax = plt.subplots(figsize=(10, 7))

    models = ['Small', 'Medium', 'Large']
    colors = ['#3498db', '#f39c12', '#e74c3c']

    powers = []
    latencies = []

    for model in ['small', 'medium', 'large']:
        powers.append(power_data[model]['avg_power_watts'])
        latencies.append(power_data[model]['avg_latency_ms'])

    # Create scatter plot
    for i, (model, color) in enumerate(zip(models, colors)):
        ax.scatter(latencies[i], powers[i], s=300, c=color, alpha=0.8,
                  edgecolors='black', linewidth=2, label=f'{model} Model')

        # Add text labels
        ax.annotate(f'{model}\n{powers[i]:.1f}W, {latencies[i]:.1f}ms',
                   (latencies[i], powers[i]),
                   textcoords="offset points",
                   xytext=(0,20), ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Average Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Power (Watts)', fontsize=13, fontweight='bold')
    ax.set_title('Power vs Latency Tradeoff', fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved: {save_path}")
    plt.close()

def plot_total_energy_comparison(save_path="results/figures/total_energy_comparison.png"):
    """Bar chart comparing total energy consumption"""
    power_data = load_power_data()

    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['Small Only', 'Medium Only', 'Large Only']
    total_energies = [power_data['small']['total_energy_joules'],
                     power_data['medium']['total_energy_joules'],
                     power_data['large']['total_energy_joules']]
    colors = ['#3498db', '#f39c12', '#e74c3c']

    bars = ax.bar(models, total_energies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, energy in zip(bars, total_energies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.1f}J\n({energy/3600:.4f}Wh)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Total Energy Consumption (Joules)', fontsize=13, fontweight='bold')
    ax.set_title('Total Energy Consumption (872 queries)', fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)

    # Add efficiency comparison text
    small_energy = total_energies[0]
    large_energy = total_energies[2]
    efficiency_gain = (large_energy - small_energy) / large_energy * 100

    ax.text(0.02, 0.98, f'Small vs Large:\n{efficiency_gain:.1f}% energy savings',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            verticalalignment='top')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved: {save_path}")
    plt.close()

def generate_all_power_visualizations():
    """Generate all power-focused visualizations"""
    print("=" * 60)
    print("GENERATING POWER CONSUMPTION VISUALIZATIONS")
    print("=" * 60)

    plot_power_comparison()
    plot_energy_efficiency()
    plot_power_vs_latency()
    plot_total_energy_comparison()

    print("\n" + "=" * 60)
    print("[DONE] ALL POWER VISUALIZATIONS GENERATED")
    print("=" * 60)
    print("\nPower plots saved to: results/figures/")
    print("  - power_comparison.png")
    print("  - energy_efficiency.png")
    print("  - power_vs_latency.png")
    print("  - total_energy_comparison.png")

if __name__ == "__main__":
    generate_all_power_visualizations()