#!/usr/bin/env python3
"""
plot_lambda_choice_boxplots.py

Legge i risultati salvati da lambda_choice.py e genera
un boxplot separato per ogni metrica (RE, PSNR, SSIM).

Output:
evaluation/results/lambda_choice/boxplots/
"""

import os
import sys
import json
from matplotlib.patches import Patch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f"Project root: {PROJECT_ROOT}")

EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation", "results", "lambda_choice")
RESULTS_FILE = os.path.join(EVAL_DIR, "results.json")  # cambia se necessario
SAVE_DIR = os.path.join(EVAL_DIR, "boxplots")


METRICS = ["RE", "PSNR", "SSIM"]


def load_results():
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}")

    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

    return results


def sort_lambda_keys(lambda_keys):
    return sorted(lambda_keys, key=lambda x: float(x))


def plot_metric_boxplot(results, metric_name, save_path):
    fig, ax = plt.subplots(figsize=(14, 6))

    data = []
    labels = []
    positions = []
    box_colors = []

    angle_keys = sorted(results.keys(), key=lambda x: int(x))

    angle_color_map = {
        "45":  "#4C72B0",
        "60":  "#55A868",
        "90":  "#C44E52",
        "180": "#8172B2",
    }

    pos = 1
    gap_between_angles = 1.5

    for angle_key in angle_keys:
        lambda_keys = sort_lambda_keys(results[angle_key].keys())

        for lambda_key in lambda_keys:
            metric_values = results[angle_key][lambda_key][metric_name]
            data.append(metric_values)
            labels.append(f"λ={float(lambda_key):.0e}")
            positions.append(pos)
            box_colors.append(angle_color_map.get(angle_key, "#999999"))
            pos += 1

        pos += gap_between_angles

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
    )

    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    for whisker in bp["whiskers"]:
        whisker.set_color("black")
        whisker.set_linewidth(1.0)

    for cap in bp["caps"]:
        cap.set_color("black")
        cap.set_linewidth(1.0)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(
        f"{metric_name} distribution across lambda values",
        fontsize=14,
        fontweight="bold"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # linee verticali per separare i gruppi di angoli
    current_pos = 1
    for angle_key in angle_keys[:-1]:
        n_lambdas = len(results[angle_key].keys())
        current_pos += n_lambdas
        ax.axvline(current_pos + gap_between_angles / 2 - 0.5,
                   color="gray", linestyle="--", alpha=0.35)
        current_pos += gap_between_angles

    # legenda colori
    legend_handles = [
        Patch(facecolor=angle_color_map[k], edgecolor="black", label=f"{k} angles")
        for k in angle_keys
    ]
    legend_loc = "upper right" if metric_name == "RE" else "upper left"

    ax.legend(
        handles=legend_handles,
        title="Projection angles",
        loc=legend_loc
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")

    
def main():
    print("Loading lambda choice results...")
    results = load_results()

    os.makedirs(SAVE_DIR, exist_ok=True)

    for metric_name in METRICS:
        save_path = os.path.join(SAVE_DIR, f"boxplot_{metric_name}.png")
        plot_metric_boxplot(results, metric_name, save_path)

    print(f"Done. Boxplots saved in: {SAVE_DIR}")


if __name__ == "__main__":
    main()