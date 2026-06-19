import torch
import os, sys
import time
import numpy as np

from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from methods.pnp.dataset import TCDataset
from methods.pnp.hqs import PnpHQS_Solver
from third_party.ippy.operators import CTProjector


PARAMS_PER_ANGOLO = {
    180: {"num_iter": 10, "mu": 0.025, "n_cg": 5},
    90:  {"num_iter": 12, "mu": 0.06, "n_cg": 5},
    60:  {"num_iter": 15, "mu": 0.08, "n_cg": 5},
    45:  {"num_iter": 15, "mu": 0.1, "n_cg": 7},
}


def process_patient(sinogram, angles, solver, num_iter=30, mu=0.1, n_cg=5):
    """
    Esegue FBP (baseline) e PnP-HQS-CG sulla stessa slice.

    """
    img_shape  = (256, 256)
    det_count  = 512

    projector = CTProjector(
        img_shape=img_shape,
        angles=angles,
        det_size=det_count,
        geometry="parallel",
        force_cpu=False
    )

    print("-> Esecuzione FBP...")
    t0 = time.time()
    with torch.no_grad():
        fbp_recon = projector.FBP(sinogram)
        fbp_recon = torch.clamp(fbp_recon, 0.0, 1.0)
    print(f"   [FBP] Fatto in {time.time()-t0:.3f}s")


    print("-> Esecuzione PnP-HQS-CG...")
    t1 = time.time()
    pnp_recon = solver.reconstruct(
        sinogram=sinogram,
        projector=projector,
        x_init=fbp_recon,    
        num_iterations=num_iter,
        mu=mu,
        n_cg=n_cg
    )
    print(f"   [PnP-CG] Fatto in {time.time()-t1:.3f}s")

    return fbp_recon, pnp_recon


# ================================================================== #
#  MAIN
# ================================================================== #
# ================================================================== #
#  RUNNER per una singola configurazione di angoli
# ================================================================== #
def run_for_angles(num_angoli, device=None):
    DEVICE = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH_PESI             = "methods/pnp/pnp_denoiser_best_v2.pth"
    CARTELLA_SINOGRAMMI   = f"data/sinogram_corrupted/test/angles_{num_angoli}"
    CARTELLA_GROUND_TRUTH = "data/preprocessed/test"
    CARTELLA_OUTPUT       = f"data/pnp/results_pnp_{num_angoli}"

    print("--- INIZIALIZZAZIONE PIPELINE PnP-HQS-CG ---")
    print(f"    Device: {DEVICE}")
    print(f"    Angoli: {num_angoli}")
    print(f"    Sinogrammi: {CARTELLA_SINOGRAMMI}")
    print(f"    Output:     {CARTELLA_OUTPUT}")

    solver = PnpHQS_Solver(denoiser_path=PATH_PESI, device=DEVICE)

    test_dataset = TCDataset(sino_dir=CARTELLA_SINOGRAMMI, gt_dir=CARTELLA_GROUND_TRUTH)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    os.makedirs(CARTELLA_OUTPUT, exist_ok=True)

    risultati = []

    for i, (my_sinogram, clean_img) in enumerate(test_loader):

        my_sinogram = my_sinogram.to(DEVICE)
        clean_img   = clean_img.to(DEVICE)

        num_angoli_slice = my_sinogram.shape[-2]
        my_angles  = np.linspace(0, np.pi, num_angoli_slice, endpoint=False)

        params = PARAMS_PER_ANGOLO.get(
            num_angoli_slice,
            {"num_iter": 15, "mu": 0.1, "n_cg": 5}  # fallback generico
        )

        print(f"\n{'='*45}")
        print(f"  Slice {i+1:3d} | angoli={num_angoli_slice} | params={params}")
        print(f"{'='*45}")

        img_fbp, img_pnp = process_patient(
            sinogram=my_sinogram,
            angles=my_angles,
            solver=solver,
            **params
        )

        gt_np  = clean_img.cpu().numpy()[0, 0]
        fbp_np = img_fbp.cpu().numpy()[0, 0]
        pnp_np = img_pnp.cpu().numpy()[0, 0]

        psnr_fbp = psnr(gt_np, fbp_np, data_range=1.0)
        psnr_pnp = psnr(gt_np, pnp_np, data_range=1.0)
        ssim_fbp = ssim(gt_np, fbp_np, data_range=1.0)
        ssim_pnp = ssim(gt_np, pnp_np, data_range=1.0)

        print(f"   [FBP]     PSNR={psnr_fbp:.2f} dB  SSIM={ssim_fbp:.4f}")
        print(f"   [PnP-CG]  PSNR={psnr_pnp:.2f} dB  SSIM={ssim_pnp:.4f}")

        risultati.append({
            "slice":    i + 1,
            "psnr_fbp": psnr_fbp, "ssim_fbp": ssim_fbp,
            "psnr_pnp": psnr_pnp, "ssim_pnp": ssim_pnp,
        })

        np.save(os.path.join(CARTELLA_OUTPUT, f"pnp_{i+1:03d}.npy"), pnp_np)

    psnr_fbp_mean = np.mean([r["psnr_fbp"] for r in risultati])
    psnr_pnp_mean = np.mean([r["psnr_pnp"] for r in risultati])
    ssim_fbp_mean = np.mean([r["ssim_fbp"] for r in risultati])
    ssim_pnp_mean = np.mean([r["ssim_pnp"] for r in risultati])

    print(f"\n{'='*45}")
    print(f"  MEDIE FINALI ({num_angoli} angoli) su {len(risultati)} slice")
    print(f"{'='*45}")
    print(f"  [FBP]     PSNR={psnr_fbp_mean:.2f} dB  SSIM={ssim_fbp_mean:.4f}")
    print(f"  [PnP-CG]  PSNR={psnr_pnp_mean:.2f} dB  SSIM={ssim_pnp_mean:.4f}")

    import csv
    csv_path = os.path.join(CARTELLA_OUTPUT, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=risultati[0].keys())
        writer.writeheader()
        writer.writerows(risultati)
    print(f"\n  Metriche salvate in: {csv_path}")
    print(f"  Immagini salvate in: {CARTELLA_OUTPUT}/")
    print(f"\nFATTO! Elaborate {len(risultati)} slice ({num_angoli} angoli).")

    return {
        "angles": num_angoli,
        "psnr_fbp": psnr_fbp_mean, "ssim_fbp": ssim_fbp_mean,
        "psnr_pnp": psnr_pnp_mean, "ssim_pnp": ssim_pnp_mean,
    }


if __name__ == "__main__":
    run_for_angles(180)