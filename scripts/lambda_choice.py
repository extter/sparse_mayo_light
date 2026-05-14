#!/usr/bin/env python3
"""
lambda_choice.py

Esegue il tuning euristico di lambda per la ricostruzione TV
su un sottoinsieme del training set, salvando:
- immagini qualitative per ogni lambda
- metriche quantitative in results.json
- boxplot finale per RE, PSNR e SSIM
"""

import os
import sys
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import create_sinogram_dataloaders
from methods.variational import (
    get_device,
    create_operators_dict,
    create_tv_solver,
    reconstruct_tv,
)
from third_party.ippy import metrics

BASE_DATA_PATH = os.path.join(PROJECT_ROOT, "data")
TV_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "tv_config.yaml")


def load_config():
    if not os.path.exists(TV_CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found: {TV_CONFIG_PATH}")

    with open(TV_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    required_keys = [
        "image_size",
        "angle_configs",
        "maxiter",
        "p",
        "batch_size",
        "lambda_sweep",
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing key '{key}' in {TV_CONFIG_PATH}")

    return config


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def format_lambda_label(lmbda):
    return f"{lmbda:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def save_boxplot_results(results, angle_configs, lambda_values, output_path):
    metrics_names = ["RE", "PSNR", "SSIM"]
    lambda_labels = [format_lambda_label(lmbda) for lmbda in lambda_values]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    n_lambdas = len(lambda_values)
    n_angles = len(angle_configs)
    width = 0.15
    spacing = 0.05

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, metric_name in zip(axes, metrics_names):
        for j, (n_ang, color) in enumerate(zip(angle_configs, colors)):
            positions = []
            data = []

            for i, lmbda in enumerate(lambda_values):
                pos = i * (n_angles * width + spacing) + j * width
                positions.append(pos)
                data.append(results[n_ang][lmbda][metric_name])

            ax.boxplot(
                data,
                positions=positions,
                widths=width * 0.85,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.7),
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=dict(marker="o", color=color, markersize=3),
            )

        group_width = n_angles * width + spacing
        tick_positions = [
            i * group_width + (n_angles - 1) * width / 2
            for i in range(n_lambdas)
        ]

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(lambda_labels, fontsize=10)
        ax.set_xlabel("λ", fontsize=12)
        ax.set_title(metric_name, fontsize=13, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7)
        for color in colors[:len(angle_configs)]
    ]
    fig.legend(
        handles,
        [f"{a} angles" for a in angle_configs],
        loc="lower center",
        ncol=len(angle_configs),
        fontsize=11,
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.suptitle("Reconstruction metrics — Chambolle-Pock TpV", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Starting TV lambda choice...")

    config = load_config()

    image_size = config["image_size"]
    angle_configs = config["angle_configs"]
    maxiter = config["maxiter"]
    p = config["p"]
    batch_size = config["batch_size"]

    lambda_values = config["lambda_sweep"]["values"]
    subset_size = config["lambda_sweep"]["subset_size"]
    save_base_dir = os.path.join(PROJECT_ROOT, config["lambda_sweep"]["save_dir"])
    results_filename = config["lambda_sweep"]["results_filename"]
    boxplot_filename = config["lambda_sweep"]["boxplot_filename"]

    device = get_device()

    print(f"Using device: {device}")
    print(
        f"image_size={image_size}, angle_configs={angle_configs}, "
        f"maxiter={maxiter}, p={p}, batch_size={batch_size}, "
        f"subset_size={subset_size}"
    )

    _, dataloaders = create_sinogram_dataloaders(
        base_data_path=BASE_DATA_PATH,
        angle_configs=angle_configs,
        batch_size=batch_size,
    )

    operators_dict = create_operators_dict(
        angle_configs=angle_configs,
        image_size=image_size,
    )

    results = {}

    for n_angles in angle_configs:
        print(f"\n===== {n_angles} ANGLES =====")

        projector = operators_dict[n_angles]
        solver = create_tv_solver(projector)
        loader = dataloaders["train"][n_angles]

        subset = []
        for i, batch in enumerate(loader):
            subset.append(batch)
            if i >= subset_size - 1:
                break

        results[n_angles] = {}

        for lmbda in lambda_values:
            print(f"\n--- lambda = {lmbda} ---")

            lmbda_str = format_lambda_label(lmbda)
            save_dir = os.path.join(
                save_base_dir,
                f"angles_{n_angles}",
                f"lambda_{lmbda_str}",
            )
            os.makedirs(save_dir, exist_ok=True)

            metrics_list = {"RE": [], "PSNR": [], "SSIM": []}

            for idx, (y_delta, y_clean, x_true, fnames) in enumerate(subset):
                y_delta = y_delta.to(device)
                x_true = x_true.to(device)

                print(f"  sample {idx}")

                x_sol, _ = reconstruct_tv(
                    y_delta=y_delta,
                    solver=solver,
                    lmbda=lmbda,
                    maxiter=maxiter,
                    p=p,
                    x_true=x_true,
                    starting_point=None,
                    verbose=False,
                )

                re = float(metrics.RE(x_sol.cpu(), x_true.cpu()))
                psnr = float(metrics.PSNR(x_sol.cpu(), x_true.cpu()))
                ssim = float(metrics.SSIM(x_sol.cpu(), x_true.cpu()))

                metrics_list["RE"].append(re)
                metrics_list["PSNR"].append(psnr)
                metrics_list["SSIM"].append(ssim)

                fig, axes = plt.subplots(1, 2, figsize=(8, 4))

                axes[0].imshow(x_true.cpu().squeeze(), cmap="gray")
                axes[0].set_title("Ground Truth")
                axes[0].axis("off")

                axes[1].imshow(x_sol.detach().cpu().squeeze(), cmap="gray")
                axes[1].set_title(
                    f"λ={lmbda_str} | {n_angles} angles\n"
                    f"RE={re:.4f} | PSNR={psnr:.2f} | SSIM={ssim:.4f}"
                )
                axes[1].axis("off")

                plt.tight_layout()
                plt.savefig(
                    os.path.join(save_dir, f"sample_{idx:03d}.png"),
                    dpi=100,
                    bbox_inches="tight",
                )
                plt.close(fig)

            results[n_angles][lmbda] = {
                "RE": metrics_list["RE"],
                "PSNR": metrics_list["PSNR"],
                "SSIM": metrics_list["SSIM"],
                "RE_mean": np.mean(metrics_list["RE"]),
                "RE_std": np.std(metrics_list["RE"]),
                "PSNR_mean": np.mean(metrics_list["PSNR"]),
                "PSNR_std": np.std(metrics_list["PSNR"]),
                "SSIM_mean": np.mean(metrics_list["SSIM"]),
                "SSIM_std": np.std(metrics_list["SSIM"]),
            }

            print(
                f"λ={lmbda} | "
                f"RE={results[n_angles][lmbda]['RE_mean']:.4f} ± "
                f"{results[n_angles][lmbda]['RE_std']:.4f}, "
                f"PSNR={results[n_angles][lmbda]['PSNR_mean']:.4f} ± "
                f"{results[n_angles][lmbda]['PSNR_std']:.4f}, "
                f"SSIM={results[n_angles][lmbda]['SSIM_mean']:.4f} ± "
                f"{results[n_angles][lmbda]['SSIM_std']:.4f}"
            )

    os.makedirs(save_base_dir, exist_ok=True)

    results_path = os.path.join(save_base_dir, results_filename)
    with open(results_path, "w") as f:
        json.dump(results, f, default=to_serializable, indent=2)

    boxplot_path = os.path.join(save_base_dir, boxplot_filename)
    save_boxplot_results(
        results=results,
        angle_configs=angle_configs,
        lambda_values=lambda_values,
        output_path=boxplot_path,
    )

    print(f"\nResults saved to: {results_path}")
    print(f"Boxplot saved to: {boxplot_path}")
    print("TV lambda choice completed!")


if __name__ == "__main__":
    main()