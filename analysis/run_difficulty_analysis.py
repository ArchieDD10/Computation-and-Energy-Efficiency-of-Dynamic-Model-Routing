"""
Difficulty Analysis Runner
Author: Archie 
Purpose: Assign 4-level sample difficulty from baseline predictions and generate visualizations.

Difficulty definition:
  Level 0 (unresolved): all models are wrong
  Level 1 (easy):       small model predicts correctly
  Level 2 (medium):     small fails, medium predicts correctly
  Level 3 (hard):       large is first model to predict correctly
"""

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11


SMALL_CSV = "results/baseline_small_only.csv"
MEDIUM_CSV = "results/baseline_medium_only.csv"
LARGE_CSV = "results/baseline_large_only.csv"
ROUTING_CSV = "results/baseline_results.csv"

DIFFICULTY_OUTPUT_CSV = "results/difficulty_levels.csv"
FIGURES_DIR = "results/figures"


def _require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Run `python analysis/run_baselines.py` first to generate single-model baselines."
        )


def _add_match_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # If text/label pairs repeat, this gives each duplicate a stable index so merges stay one-to-one.
    out["_dup_id"] = out.groupby(["text", "true_label"]).cumcount()
    return out


def load_required_baselines() -> Dict[str, pd.DataFrame]:
    _require_file(SMALL_CSV)
    _require_file(MEDIUM_CSV)
    _require_file(LARGE_CSV)

    return {
        "small": pd.read_csv(SMALL_CSV),
        "medium": pd.read_csv(MEDIUM_CSV),
        "large": pd.read_csv(LARGE_CSV),
    }


def align_baselines(baselines: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    small_df = baselines["small"].copy()
    medium_df = baselines["medium"].copy()
    large_df = baselines["large"].copy()

    same_length = len(small_df) == len(medium_df) == len(large_df)
    same_labels = (
        same_length
        and small_df["true_label"].equals(medium_df["true_label"])
        and small_df["true_label"].equals(large_df["true_label"])
    )

    if same_labels:
        # Fast path: most runs keep the same ordering across all baseline CSVs.
        aligned = pd.DataFrame(
            {
                "sample_id": small_df.index,
                "text": small_df["text"],
                "true_label": small_df["true_label"],
                "pred_small": small_df["pred_label"],
                "pred_medium": medium_df["pred_label"],
                "pred_large": large_df["pred_label"],
                "confidence_small": small_df["confidence"],
                "confidence_medium": medium_df["confidence"],
                "confidence_large": large_df["confidence"],
                "latency_small_ms": small_df["latency_ms"],
                "latency_medium_ms": medium_df["latency_ms"],
                "latency_large_ms": large_df["latency_ms"],
            }
        )
        return aligned

    # Fallback for rare ordering mismatch: align by text + true_label + duplicate index.
    s = _add_match_key(small_df).rename(
        columns={
            "pred_label": "pred_small",
            "confidence": "confidence_small",
            "latency_ms": "latency_small_ms",
        }
    )
    m = _add_match_key(medium_df).rename(
        columns={
            "pred_label": "pred_medium",
            "confidence": "confidence_medium",
            "latency_ms": "latency_medium_ms",
        }
    )
    l = _add_match_key(large_df).rename(
        columns={
            "pred_label": "pred_large",
            "confidence": "confidence_large",
            "latency_ms": "latency_large_ms",
        }
    )

    merged = s.merge(
        m[
            [
                "text",
                "true_label",
                "_dup_id",
                "pred_medium",
                "confidence_medium",
                "latency_medium_ms",
            ]
        ],
        on=["text", "true_label", "_dup_id"],
        how="inner",
    ).merge(
        l[
            [
                "text",
                "true_label",
                "_dup_id",
                "pred_large",
                "confidence_large",
                "latency_large_ms",
            ]
        ],
        on=["text", "true_label", "_dup_id"],
        how="inner",
    )

    merged = merged.reset_index(drop=True)
    merged["sample_id"] = merged.index
    return merged[
        [
            "sample_id",
            "text",
            "true_label",
            "pred_small",
            "pred_medium",
            "pred_large",
            "confidence_small",
            "confidence_medium",
            "confidence_large",
            "latency_small_ms",
            "latency_medium_ms",
            "latency_large_ms",
        ]
    ]


def assign_difficulty_levels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Per-model correctness flags used for both labeling and later visualizations.
    out["small_correct"] = out["pred_small"] == out["true_label"]
    out["medium_correct"] = out["pred_medium"] == out["true_label"]
    out["large_correct"] = out["pred_large"] == out["true_label"]
    out["all_models_wrong"] = ~(
        out["small_correct"] | out["medium_correct"] | out["large_correct"]
    )

    # "First success" mirrors escalation order: small -> medium -> large.
    # all_models_wrong is explicitly bucketed as "none" (Level 0).
    out["first_success_model"] = "large"
    out.loc[out["small_correct"], "first_success_model"] = "small"
    out.loc[(~out["small_correct"]) & out["medium_correct"], "first_success_model"] = "medium"
    out.loc[out["all_models_wrong"], "first_success_model"] = "none"

    out["difficulty_level"] = out["first_success_model"].map(
        {"none": 0, "small": 1, "medium": 2, "large": 3}
    )

    # Reconstruct expected routing cost for each sample from baseline latencies.
    # Level 0 and Level 3 both imply the request reached the large model.
    out["estimated_routing_latency_ms"] = (
        out["latency_small_ms"] + out["latency_medium_ms"] + out["latency_large_ms"]
    )
    out.loc[out["difficulty_level"] == 1, "estimated_routing_latency_ms"] = out["latency_small_ms"]
    out.loc[out["difficulty_level"] == 2, "estimated_routing_latency_ms"] = (
        out["latency_small_ms"] + out["latency_medium_ms"]
    )

    return out


def _ensure_figures_dir() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_difficulty_distribution(df: pd.DataFrame) -> None:
    _ensure_figures_dir()
    # First overview plot: how much of the dataset falls into each difficulty bucket.
    level_counts = df["difficulty_level"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
    percentages = level_counts / len(df) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        [f"Level {lvl}" for lvl in level_counts.index],
        level_counts.values,
        color=["#7f8c8d", "#3498db", "#f39c12", "#e74c3c"],
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
    )

    for bar, count, pct in zip(bars, level_counts.values, percentages.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Number of Samples", fontsize=12, fontweight="bold")
    ax.set_title("Difficulty Level Distribution (0-3)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    save_path = os.path.join(FIGURES_DIR, "difficulty_level_distribution.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Saved: {save_path}")


def plot_model_accuracy_by_level(df: pd.DataFrame) -> None:
    _ensure_figures_dir()

    rows = []
    for level in [0, 1, 2, 3]:
        group = df[df["difficulty_level"] == level]
        if len(group) == 0:
            continue
        rows.extend(
            [
                {"difficulty_level": level, "model": "small", "accuracy": group["small_correct"].mean() * 100},
                {"difficulty_level": level, "model": "medium", "accuracy": group["medium_correct"].mean() * 100},
                {"difficulty_level": level, "model": "large", "accuracy": group["large_correct"].mean() * 100},
            ]
        )

    # Long-form table is convenient for seaborn grouped bar plotting.
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=plot_df,
        x="difficulty_level",
        y="accuracy",
        hue="model",
        palette={"small": "#3498db", "medium": "#f39c12", "large": "#e74c3c"},
        ax=ax,
    )

    ax.set_xlabel("Difficulty Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Model Accuracy Within Each Difficulty Level", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)

    save_path = os.path.join(FIGURES_DIR, "model_accuracy_by_difficulty_level.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Saved: {save_path}")


def plot_estimated_latency_by_level(df: pd.DataFrame) -> None:
    _ensure_figures_dir()

    # Shows how escalation cost increases as samples get harder.
    latency_summary = (
        df.groupby("difficulty_level")["estimated_routing_latency_ms"]
        .mean()
        .reindex([0, 1, 2, 3])
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        [f"Level {idx}" for idx in latency_summary.index],
        latency_summary.values,
        color=["#7f8c8d", "#3498db", "#f39c12", "#e74c3c"],
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
    )

    for bar, val in zip(bars, latency_summary.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.1f}ms",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Estimated Routing Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Estimated Routing Latency by Difficulty Level", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    save_path = os.path.join(FIGURES_DIR, "estimated_latency_by_difficulty_level.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Saved: {save_path}")


def load_optional_routing_results() -> Optional[pd.DataFrame]:
    if not os.path.exists(ROUTING_CSV):
        return None
    return pd.read_csv(ROUTING_CSV)


def plot_routing_vs_difficulty(df: pd.DataFrame, routing_df: Optional[pd.DataFrame]) -> None:
    if routing_df is None or "chosen_model" not in routing_df.columns:
        print("[INFO] Routing comparison plot skipped (routing file missing or incompatible).")
        return

    if len(routing_df) != len(df):
        # We only do index-based comparison when both files represent the same sample set.
        print("[INFO] Routing comparison plot skipped (row count mismatch).")
        return

    merged = pd.DataFrame(
        {
            "difficulty_level": df["difficulty_level"].values,
            "chosen_model": routing_df["chosen_model"].values,
        }
    )
    ctab = pd.crosstab(
        merged["difficulty_level"], merged["chosen_model"], normalize="index"
    ).reindex(index=[0, 1, 2, 3], fill_value=0)

    # Convert proportions to percentages for easier reading on stacked bars.
    ctab = ctab * 100.0

    fig, ax = plt.subplots(figsize=(11, 6))
    ctab.plot(
        kind="bar",
        stacked=True,
        color={"small": "#3498db", "medium": "#f39c12", "large": "#e74c3c"},
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_xlabel("Difficulty Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("Routing Model Usage (%)", fontsize=12, fontweight="bold")
    ax.set_title("Routing Model Choice by Difficulty Level", fontsize=14, fontweight="bold")
    ax.legend(title="Chosen Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    save_path = os.path.join(FIGURES_DIR, "routing_model_choice_by_difficulty.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Saved: {save_path}")


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    level_dist = df["difficulty_level"].value_counts().reindex([0, 1, 2, 3], fill_value=0)

    print("\n" + "=" * 70)
    print("DIFFICULTY SUMMARY")
    print("=" * 70)
    for level in [0, 1, 2, 3]:
        count = int(level_dist.get(level, 0))
        pct = (count / total * 100.0) if total else 0.0
        print(f"Level {level}: {count:4d} samples ({pct:5.1f}%)")

    all_wrong = int(df["all_models_wrong"].sum())
    print(f"\nAll models wrong: {all_wrong} samples ({all_wrong / total * 100:.1f}%)")

    print("\nAverage estimated routing latency by level:")
    for level, avg_ms in (
        df.groupby("difficulty_level")["estimated_routing_latency_ms"].mean().items()
    ):
        print(f"  Level {level}: {avg_ms:.2f} ms")


def main() -> None:
    print("=" * 70)
    print("DIFFICULTY ANALYSIS (LEVELS 0-3)")
    print("=" * 70)

    # Pipeline: load -> align rows -> assign difficulty labels.
    baselines = load_required_baselines()
    aligned = align_baselines(baselines)
    labeled = assign_difficulty_levels(aligned)

    os.makedirs("results", exist_ok=True)
    labeled.to_csv(DIFFICULTY_OUTPUT_CSV, index=False)
    print(f"\n[DONE] Saved: {DIFFICULTY_OUTPUT_CSV}")

    plot_difficulty_distribution(labeled)
    plot_model_accuracy_by_level(labeled)
    plot_estimated_latency_by_level(labeled)
    plot_routing_vs_difficulty(labeled, load_optional_routing_results())

    print_summary(labeled)

    print("\n" + "=" * 70)
    print("[DONE] Difficulty analysis complete.")
    print("Figures saved in: results/figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
