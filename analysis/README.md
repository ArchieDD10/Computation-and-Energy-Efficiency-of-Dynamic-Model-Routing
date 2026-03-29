# Analysis Module

**Author:** Archie Deguzman
**Purpose:** Latency measurement, power consumption tracking, baseline comparison, and visualization for dynamic routing research

---

## Overview

This module provides analysis tools that work with Kamran's routing code without modifying it. my responsibilities:

1. ✅ **Measure latency** for individual models (no routing)
2. ✅ **Track power consumption** using GPU sensors (NVIDIA NVML)
3. ✅ **Compare** routing vs single-model baselines
4. ✅ **Visualize** results with publication-ready graphs

---

## Power Monitoring Setup

### Install NVIDIA Power Monitoring Library

```bash
pip install nvidia-ml-py3
```

### Test Power Monitoring

```bash
python analysis/power_monitor.py
```

Expected output:
```
POWER MONITORING TEST
[INFO] NVIDIA GPU power monitoring enabled
Monitoring GPU power for 3 seconds...

Power Statistics:
  Samples collected: 300
  Duration: 3.00 seconds
  Average power: 85.42 W
  Max power: 120.15 W
  Min power: 45.23 W
  Total energy: 256.26 J (0.000712 Wh)
```

### Run Baselines with Power Monitoring

```bash
python analysis/run_baselines_with_power.py
```

This measures **both latency and power consumption** for each model.

---

## Quick Start

### Option 1: Run Everything at Once

```bash
python analysis/run_all.py
```

This runs:
- Single-model baselines (small, medium, large)
- Comparison analysis
- All visualizations

### Option 2: Step-by-Step

```bash
# Step 1: Run single-model baselines
python analysis/run_baselines.py

# Step 2: Compare routing vs baselines
python analysis/compare_latency.py

# Step 3: Generate visualizations
python analysis/visualize.py
```

---

## Files in This Module

### `run_baselines.py`
Runs each model individually (no routing) to establish baseline performance.

**Creates:**
- `results/baseline_small_only.csv`
- `results/baseline_medium_only.csv`
- `results/baseline_large_only.csv`

**Metrics collected:**
- Per-query latency
- Accuracy
- Confidence scores

### `compare_latency.py`
Compares routing strategy against single-model baselines.

**Creates:**
- `results/comparison_summary.csv`

**Metrics computed:**
- Average latency per approach
- Total latency savings
- Speedup vs always-large baseline
- Model usage distribution (for routing)

### `visualize.py`
Generates 6 publication-quality graphs:

1. **Latency Comparison** - Bar chart of avg latency
2. **Accuracy vs Latency** - Scatter plot showing tradeoffs
3. **Model Usage Distribution** - Pie chart (routing only)
4. **Latency Distribution** - Violin plots comparing distributions
5. **Cumulative Latency** - Line plot over time
6. **Speedup Comparison** - Bar chart vs baseline

**Creates:**
- `results/figures/*.png` (6 graphs)

### `run_all.py`
Master script that runs everything in sequence.

---

## Workflow

### Kamran's Code
```
scripts/
├── routing.py          (Escalation router logic)
└── run_baseline.py     (Runs routing experiment)
```

**Output:** `results/baseline_results.csv` (routing results)

### My Code (This Module)
```
analysis/
├── run_baselines.py    (Single-model baselines)
├── compare_latency.py  (Comparison analysis)
├── visualize.py        (Graph generation)
└── run_all.py          (Master script)
```

**Outputs:**
- `results/baseline_*_only.csv` (baseline data)
- `results/comparison_summary.csv` (metrics)
- `results/figures/*.png` (graphs)

---

## Prerequisites

Your partner must run their code first:
```bash
cd scripts
python run_baseline.py
```

This creates `results/baseline_results.csv` with routing data.

Then you can run your analysis module.

---

## Example Output

### Latency Comparison Table
```
Approach              Accuracy   Avg Latency   Total Latency      Speedup
----------------------------------------------------------------------------------
Always Small            0.9174      5.55ms       4839.6ms        3.00x
Always Medium           0.9243     12.43ms      10838.4ms        1.34x
Always Large            0.9323     16.65ms      14519.8ms        1.00x
Routing (Escalation)    0.9300     10.25ms       8938.0ms        1.62x
```

### Model Usage (Routing)
```
Small model:   650 queries (74.5%)
Medium model:  150 queries (17.2%)
Large model:    72 queries ( 8.3%)

Average models invoked per query: 1.34
```

---

## Customization

### Change Model Sizes for Graphs
Edit `visualize.py` and modify the color scheme:
```python
colors = {
    'small': '#3498db',   # Blue
    'medium': '#f39c12',  # Orange
    'large': '#e74c3c',   # Red
    'routing': '#2ecc71'  # Green
}
```

### Add New Metrics
Edit `compare_latency.py` → `compute_metrics()` function.

### Add New Graphs
Edit `visualize.py` → Add new `plot_*()` function.

---

## Troubleshooting

### Error: "Routing results not found"
Your partner needs to run: `python scripts/run_baseline.py`

### Error: "Baseline not found"
Run: `python analysis/run_baselines.py`

### Empty Figures Directory
Run: `python analysis/visualize.py`

### Models Not Loading
Check that `local_models/` directory exists and contains:
- `distilbert-base-uncased-finetuned-sst-2-english/`
- `textattack__bert-base-uncased-SST-2/`
- `siebert__sentiment-roberta-large-english/`

---

## Tips for Your Research

1. **Run baselines regularly** - Whenever your partner changes thresholds, re-run baselines
2. **Version your results** - Save figures with timestamped names for different experiments
3. **Document findings** - Add notes to `comparison_summary.csv` about what worked
4. **Compare threshold sweeps** - Run multiple routing experiments with different τ values

---

## Integration with Partner's Code

This module **imports** but doesn't modify:
```python
# From partner's code (read-only)
from scripts.routing import small, medium, large, normalize_label
```

If Kam changes his code, the analysis automatically uses the latest version.

---

## Questions?

This module is independent but complementary to Kam's routing code. You can:
- Run it anytime after `results/baseline_results.csv` exists
- Modify visualization styles without affecting routing logic
- Add new metrics without touching their code
