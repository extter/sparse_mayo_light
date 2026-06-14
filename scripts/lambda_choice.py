#!/usr/bin/env python3
"""
lambda_choice.py

Esegue il tuning euristico di lambda per la ricostruzione TV
su un sottoinsieme del training set, salvando:
- immagini qualitative per ogni lambda in save_dir (da config)
- metriche quantitative in evaluation/results/lambda_choice/
"""

import os
import sys
import json
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
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
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation", "results", "lambda_choice")


def load_config():
    if not os.path.exists(TV_CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found: {TV_CONFIG_PATH}")

    with open(TV_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    required_keys = [
        "image_size",
        "det_size",
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


def save_image(x_true, x_sol, lmbda_str, n_angles, re, psnr, ssim, save_path):
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
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Starting TV lambda choice...")

    config = load_config()

    image_size = config["image_size"]
    det_size = config["det_size"]
    angle_configs = config["angle_configs"]
    maxiter = config["maxiter"]
    p = config["p"]
    batch_size = config["batch_size"]

    lambda_values = config["lambda_sweep"]["values"]
    subset_size = config["lambda_sweep"]["subset_size"]
    images_base_dir = os.path.join(PROJECT_ROOT, config["lambda_sweep"]["save_dir"])
    results_filename = config["lambda_sweep"]["results_filename"]

    os.makedirs(images_base_dir, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    device = get_device()

    print(f"Using device: {device}")
    print(
        f"image_size={image_size}, det_size={det_size}, angle_configs={angle_configs}, "
        f"maxiter={maxiter}, p={p}, batch_size={batch_size}, "
        f"subset_size={subset_size}"
    )
    print(f"Images  -> {images_base_dir}")
    print(f"Metrics -> {EVAL_DIR}")

    _, dataloaders = create_sinogram_dataloaders(
        base_data_path=BASE_DATA_PATH,
        angle_configs=angle_configs,
        batch_size=batch_size,
    )

    operators_dict = create_operators_dict(
        angle_configs=angle_configs,
        image_size=image_size,
        det_size=det_size,
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

            img_save_dir = os.path.join(
                images_base_dir,
                f"angles_{n_angles}",
                f"lambda_{lmbda_str}",
            )
            os.makedirs(img_save_dir, exist_ok=True)

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

                save_image(
                    x_true=x_true,
                    x_sol=x_sol,
                    lmbda_str=lmbda_str,
                    n_angles=n_angles,
                    re=re,
                    psnr=psnr,
                    ssim=ssim,
                    save_path=os.path.join(img_save_dir, f"sample_{idx:03d}.png"),
                )

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

    results_path = os.path.join(EVAL_DIR, results_filename)
    with open(results_path, "w") as f:
        json.dump(results, f, default=to_serializable, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("TV lambda choice completed!")


if __name__ == "__main__":
    main()