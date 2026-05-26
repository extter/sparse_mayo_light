#!/usr/bin/env python3
"""
run_tv.py

Genera le ricostruzioni TV finali per tutti gli split del dataset
usando i best lambda selezionati per ciascuna configurazione angolare.
"""

import os
import sys
import yaml
import numpy as np
from tqdm import tqdm

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

SPLITS = ["train", "validation", "test"]


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
        "best_lambdas",
        "reconstruction",
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing key '{key}' in {TV_CONFIG_PATH}")

    return config


def check_output_directories(angle_configs, reco_base_path):
    for split in SPLITS:
        for n_angles in angle_configs:
            save_dir = os.path.join(reco_base_path, f"angles_{n_angles}", split)
            os.makedirs(save_dir, exist_ok=True)


def main():
    print("Starting TV reconstruction pipeline...")

    config = load_config()

    image_size = config["image_size"]
    det_size = config["det_size"]
    angle_configs = config["angle_configs"]
    maxiter = config["maxiter"]
    p = config["p"]
    batch_size = config["batch_size"]
    best_lambdas = config["best_lambdas"]
    reco_base_path = os.path.join(PROJECT_ROOT, config["reconstruction"]["save_dir"])

    device = get_device()

    print(f"Using device: {device}")
    print(
        f"image_size={image_size}, det_size={det_size}, angle_configs={angle_configs}, "
        f"maxiter={maxiter}, p={p}, batch_size={batch_size}"
    )

    check_output_directories(angle_configs, reco_base_path)

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
        print(f"\n===== {n_angles} ANGLES =====")

        if n_angles not in best_lambdas and str(n_angles) not in best_lambdas:
            raise KeyError(f"Missing best lambda for angle configuration: {n_angles}")

        best_lambda = best_lambdas.get(n_angles, best_lambdas.get(str(n_angles)))

        projector = operators_dict[n_angles]
        solver = create_tv_solver(projector)

        for split in SPLITS:
            print(f"  -- split: {split} --")
            save_dir = os.path.join(reco_base_path, f"angles_{n_angles}", split)
            loader = dataloaders[split][n_angles]

            for y_delta, y_clean, x_true, fnames in tqdm(
                loader,
                desc=f"{n_angles}a/{split}"
            ):
                y_delta = y_delta.to(device)
                x_true = x_true.to(device)

                x_sol, _ = reconstruct_tv(
                    y_delta=y_delta,
                    solver=solver,
                    lmbda=best_lambda,
                    maxiter=maxiter,
                    p=p,
                    x_true=x_true,
                    starting_point=None,
                    verbose=False,
                )

                x_sol_np = x_sol.detach().cpu().squeeze().numpy().astype(np.float32)
                np.save(os.path.join(save_dir, fnames[0]), x_sol_np)

    print("\nDone! TV reconstructions saved successfully.")


if __name__ == "__main__":
    main()