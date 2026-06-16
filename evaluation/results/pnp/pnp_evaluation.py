#!/usr/bin/env python3
"""
evaluate_pnp.py
Calcola PSNR, SSIM, RE e MSE per le ricostruzioni PnP (data/pnp/results_pnp_{n_angles}/)
confrontate con il ground truth preprocessato (data/preprocessed/test/).
Accoppiamento per indice: pnp_001.npy <-> test_C081_0.npy, ecc.
"""

import os
import sys
import numpy as np
from glob import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
print(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ANGLE_CONFIGS = [180, 90, 60, 45]
PNP_BASE_DIR  = os.path.join(PROJECT_ROOT, "data", "pnp")
GT_DIR        = os.path.join(PROJECT_ROOT, "data", "preprocessed", "test")


def relative_error(x_true, x_pred):
    return np.linalg.norm(x_pred - x_true) / np.linalg.norm(x_true)


def evaluate(pnp_dir, gt_dir, data_range=1.0):
    pnp_paths = sorted(glob(os.path.join(pnp_dir, "*.npy")))
    gt_paths  = sorted(glob(os.path.join(gt_dir,  "*.npy")))

    if not pnp_paths:
        print(f"  [WARNING] Nessun file trovato in {pnp_dir}")
        return None

    if len(pnp_paths) != len(gt_paths):
        print(
            f"  [WARNING] Numero file diverso: "
            f"{len(pnp_paths)} PnP vs {len(gt_paths)} GT"
        )

    n = min(len(pnp_paths), len(gt_paths))
    metrics = {"psnr": [], "ssim": [], "re": [], "mse": []}

    for i in range(n):
        pnp_path = pnp_paths[i]
        gt_path  = gt_paths[i]

        x_pred = np.load(pnp_path).astype(np.float32)
        x_true = np.load(gt_path).astype(np.float32)

        # squeeze per sicurezza (es. shape (1,256,256) -> (256,256))
        x_pred = x_pred.squeeze()
        x_true = x_true.squeeze()

        assert x_pred.shape == x_true.shape, (
            f"Shape mismatch: {x_pred.shape} vs {x_true.shape} "
            f"({os.path.basename(pnp_path)} <-> {os.path.basename(gt_path)})"
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
    print("=" * 50)
    print("  PnP Evaluation vs Preprocessed Ground Truth")
    print("=" * 50)

    all_results = {}

    for n_angles in ANGLE_CONFIGS:
        pnp_dir = os.path.join(PNP_BASE_DIR, f"results_pnp_{n_angles}")

        if not os.path.isdir(pnp_dir):
            print(f"\n  [SKIP] Directory non trovata: {pnp_dir}")
            continue

        metrics = evaluate(pnp_dir, GT_DIR)

        if metrics is None:
            continue

        all_results[n_angles] = metrics
        print_metrics(n_angles, metrics)

    # Riepilogo compatto
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