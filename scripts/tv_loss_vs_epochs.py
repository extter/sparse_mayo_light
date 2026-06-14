#!/usr/bin/env python3
"""
tv_loss_vs_epochs.py

Esegue la ricostruzione TV (Chambolle-Pock) su 1 immagine per ogni
configurazione angolare, loggando la loss per ogni iterazione.
Salva 4 plot separati (uno per angolo) in evaluation/results/tv_loss_vs_epochs/
"""

import os
import sys
import numpy
import yaml
import matplotlib
import numpy as np
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

BASE_DATA_PATH = os.path.join(PROJECT_ROOT, "data")
TV_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "tv_config.yaml")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation", "results", "tv_loss_vs_epochs")

MAX_ITER = 200

LAMBDA_PER_ANGLES = {
    45:  0.1, 
    60:  0.1, 
    90:  0.3, 
    180: 0.5, 

}
# ------------------------------------------------------------------ #


def load_config():
    if not os.path.exists(TV_CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found: {TV_CONFIG_PATH}")
    with open(TV_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    required_keys = ["image_size", "det_size", "angle_configs", "p", "batch_size"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing key '{key}' in {TV_CONFIG_PATH}")
    return config

def plot_loss(loss_values, n_angles, lmbda, save_path):
    print("DEBUG KEYS:", list(loss_values.keys()))

    re_values = loss_values["RE"]
    ssim_values = loss_values["SSIM"]

    if hasattr(re_values, "detach"):
        re_values = re_values.detach().cpu().numpy()
    if hasattr(ssim_values, "detach"):
        ssim_values = ssim_values.detach().cpu().numpy()

    n = len(re_values)
    iters = np.arange(1, n + 1)   # [1, 2, ..., n]

    # Se RE/SSIM hanno forma (n, 1), spremi a (n,)
    re_values = re_values.squeeze()
    ssim_values = ssim_values.squeeze()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(iters, re_values, color="#4C72B0", linewidth=1.5)
    ax1.set_xlabel("Iteration", fontsize=13)
    ax1.set_ylabel("RE", fontsize=13)
    ax1.set_title("Relative Error", fontsize=13, fontweight="bold")
    ax1.grid(linestyle="--", alpha=0.5)

    ax2.plot(iters, ssim_values, color="#DD8452", linewidth=1.5)
    ax2.set_xlabel("Iteration", fontsize=13)
    ax2.set_ylabel("SSIM", fontsize=13)
    ax2.set_title("SSIM", fontsize=13, fontweight="bold")
    ax2.grid(linestyle="--", alpha=0.5)

    fig.suptitle(
        f"Chambolle-Pock — {n_angles} angles | λ={lmbda:.0e}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved -> {save_path}")


def main():
    print("Starting TV loss vs epochs analysis...")
    print(f"max_iter = {MAX_ITER}")
    for n_ang, lmbda in LAMBDA_PER_ANGLES.items():
        print(f"  {n_ang} angles -> λ = {lmbda:.0e}")

    config = load_config()

    image_size = config["image_size"]
    det_size = config["det_size"]
    angle_configs = config["angle_configs"]
    p = config["p"]
    batch_size = config["batch_size"]

    for n_ang in angle_configs:
        if n_ang not in LAMBDA_PER_ANGLES:
            raise KeyError(
                f"Nessun lambda definito per {n_ang} angoli in LAMBDA_PER_ANGLES"
            )

    os.makedirs(EVAL_DIR, exist_ok=True)

    device = get_device()
    print(f"\nDevice: {device}")

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

    for n_angles in angle_configs:
        lmbda = LAMBDA_PER_ANGLES[n_angles]
        print(f"\n===== {n_angles} ANGLES  |  λ={lmbda:.0e} =====")

        projector = operators_dict[n_angles]
        solver = create_tv_solver(projector)
        loader = dataloaders["train"][n_angles]

        y_delta, y_clean, x_true, fnames = next(iter(loader))
        y_delta = y_delta.to(device)
        x_true = x_true.to(device)

        print(f"  Sample: {fnames[0]}")
        print(f"  Running {MAX_ITER} iterations...")

        _, loss_values = reconstruct_tv(
            y_delta=y_delta,
            solver=solver,
            lmbda=lmbda,
            maxiter=MAX_ITER,
            p=p,
            x_true=x_true,
            starting_point=None,
            verbose=True,
        )

        save_path = os.path.join(EVAL_DIR, f"loss_angles_{n_angles}.png")
        plot_loss(
            loss_values=loss_values,
            n_angles=n_angles,
            lmbda=lmbda,
            save_path=save_path,
        )

    print(f"\nDone. Plots saved in: {EVAL_DIR}")


if __name__ == "__main__":
    main()