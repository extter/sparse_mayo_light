#!/usr/bin/env python3
"""
plot_metrics_boxplots.py

Boxplot UNet: confronto tra numeri di angoli, un file per metrica.
Output: evaluation/results/end_to_end_2/boxplots/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOSS_NAME    = "mixed"
CSV_PATH     = PROJECT_ROOT / "end_to_end_2" / f"metrics_unet_{LOSS_NAME}.csv"
SAVE_DIR     = PROJECT_ROOT / "end_to_end_2" / "boxplots"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

ANGLE_CONFIGS = [45, 60, 90, 180]

METRICS = [
    ("unet_ssim", "SSIM",      "higher is better"),
    ("unet_psnr", "PSNR (dB)", "higher is better"),
    ("unet_mse",  "MSE",       "lower is better"),
    ("unet_re",   "RE",        "lower is better"),
]

ANGLE_COLOR_MAP = {
    45:  "#4C72B0",
    60:  "#55A868",
    90:  "#C44E52",
    180: "#8172B2",
}

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV non trovato: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows — angles: {sorted(df['n_angles'].unique())}")

for col, metric_label, direction in METRICS:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"UNet — {metric_label}  ({direction})", fontweight="bold")

    data      = [df[df["n_angles"] == a][col].values for a in ANGLE_CONFIGS]
    colors    = [ANGLE_COLOR_MAP[a] for a in ANGLE_CONFIGS]
    positions = list(range(1, len(ANGLE_CONFIGS) + 1))

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, linestyle="none", alpha=0.5),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)

    for pos, data_arr in zip(positions, data):
        med = np.median(data_arr)
        fmt = ".5f" if col in ("unet_mse", "unet_re") else ".4f"
        ax.text(pos, med, f"{med:{fmt}}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="black")

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{a} angles" for a in ANGLE_CONFIGS])
    ax.set_ylabel(metric_label)
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path = SAVE_DIR / f"boxplot_{col}_{LOSS_NAME}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

print(f"\nDone. Boxplots in: {SAVE_DIR}")