#!/usr/bin/env python3
"""
evaluate_tv.py
Calcola PSNR, SSIM, RE e MSE per le ricostruzioni TV sul set di test.
"""

import os
import sys
import yaml
import numpy as np
from glob import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TV_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "tv_config.yaml")
BASE_DATA_PATH  = os.path.join(PROJECT_ROOT, "data")

def load_config():
    with open(TV_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def relative_error(x_true, x_pred):
    return np.linalg.norm(x_pred - x_true) / np.linalg.norm(x_true)

def evaluate_split(reco_dir, gt_dir, data_range=1.0):
    """
    Accoppia reco e ground truth per nome file e calcola le metriche.
    Returns: dict con liste di valori per PSNR, SSIM, RE, MSE
    """
    reco_paths = sorted(glob(os.path.join(reco_dir, "*.npy")))
    if not reco_paths:
        print(f"  [WARNING] Nessun file trovato in {reco_dir}")
        return None

    metrics = {"psnr": [], "ssim": [], "re": [], "mse": []}

    for reco_path in reco_paths:
        fname = os.path.basename(reco_path)
        gt_path = os.path.join(gt_dir, fname)

        if not os.path.exists(gt_path):
            print(f"  [WARNING] GT non trovata per {fname}, skip.")
            continue

        x_pred = np.load(reco_path).astype(np.float32)
        x_true = np.load(gt_path).astype(np.float32)

        assert x_pred.shape == x_true.shape, (
            f"Shape mismatch: {x_pred.shape} vs {x_true.shape} ({fname})"
        )

        metrics["psnr"].append(psnr(x_true, x_pred, data_range=data_range))
        metrics["ssim"].append(ssim(x_true, x_pred, data_range=data_range))
        metrics["re"].append(relative_error(x_true, x_pred))
        metrics["mse"].append(np.mean((x_true - x_pred) ** 2))

    return metrics

def print_metrics(n_angles, metrics):
    n = len(metrics["psnr"])
    print(f"\n  Angoli: {n_angles}  |  Immagini valutate: {n}")
    print(f"  {'Metrica':<8}  {'Media':>10}  {'Std':>10}")
    print(f"  {'-'*32}")
    for key in ["psnr", "ssim", "re", "mse"]:
        vals = np.array(metrics[key])
        print(f"  {key.upper():<8}  {vals.mean():>10.4f}  {vals.std():>10.4f}")

def main():
    config = load_config()
    angle_configs   = config["angle_configs"]
    reco_base_path  = os.path.join(PROJECT_ROOT, config["reconstruction"]["save_dir"])
    gt_dir          = os.path.join(BASE_DATA_PATH, "preprocessed", "test")

    print("=" * 50)
    print("  TV Reconstruction — Evaluation on TEST set")
    print("=" * 50)

    all_results = {}

    for n_angles in angle_configs:
        reco_dir = os.path.join(reco_base_path, f"angles_{n_angles}", "test")
        metrics  = evaluate_split(reco_dir, gt_dir)

        if metrics is None:
            continue

        all_results[n_angles] = metrics
        print_metrics(n_angles, metrics)

    # Riepilogo compatto per la tabella LaTeX
    print("\n" + "=" * 50)
    print("  RIEPILOGO (media ± std)")
    print("=" * 50)
    print(f"  {'Angles':<8} {'PSNR':>16} {'SSIM':>16} {'RE':>16} {'MSE':>16}")
    print(f"  {'-'*72}")
    for n_angles, metrics in all_results.items():
        psnr_v = np.array(metrics["psnr"])
        ssim_v = np.array(metrics["ssim"])
        re_v   = np.array(metrics["re"])
        mse_v  = np.array(metrics["mse"])
        print(
            f"  {n_angles:<8}"
            f" {psnr_v.mean():.2f} ± {psnr_v.std():.2f}"
            f"  {ssim_v.mean():.4f} ± {ssim_v.std():.4f}"
            f"  {re_v.mean():.4f} ± {re_v.std():.4f}"
            f"  {mse_v.mean():.6f} ± {mse_v.std():.6f}"
        )

if __name__ == "__main__":
    main()