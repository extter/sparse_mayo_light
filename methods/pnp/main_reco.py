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

# ------------------------------------------------------------------ #
# Parametri HQS-CG per ciascuna configurazione di angoli.             #
# step_size e ATA_norm non servono più — rimangono solo mu e n_cg.    #
# Valori di partenza ragionevoli: affinali empiricamente.             #
# ------------------------------------------------------------------ #
PARAMS_PER_ANGOLO = {
    180: {"num_iter": 10, "mu": 0.025, "n_cg": 5},
    90:  {"num_iter": 12, "mu": 0.06, "n_cg": 5},
    60:  {"num_iter": 15, "mu": 0.08, "n_cg": 5},
    45:  {"num_iter": 15, "mu": 0.1, "n_cg": 7},
}


def process_patient(sinogram, angles, solver, num_iter=30, mu=0.1, n_cg=5):
    """
    Esegue FBP (baseline) e PnP-HQS-CG sulla stessa slice.

    La FBP viene calcolata una volta sola qui e passata come x_init
    al solver, evitando di ricalcolarla internamente (fix rispetto
    alla versione precedente che la calcolava due volte).
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

    # --- Baseline FBP (calcolata una volta sola) ---
    print("-> Esecuzione FBP...")
    t0 = time.time()
    with torch.no_grad():
        fbp_recon = projector.FBP(sinogram)
        fbp_recon = torch.clamp(fbp_recon, 0.0, 1.0)
    print(f"   [FBP] Fatto in {time.time()-t0:.3f}s")

    # --- PnP-HQS con CG interno ---
    # Parametri rispetto alla versione GD:
    #   - step_size  : RIMOSSO (il CG non ne ha bisogno)
    #   - ATA_norm   : RIMOSSO (gestito internamente dal CG)
    #   - n_inner    : SOSTITUITO da n_cg (iterazioni Conjugate Gradient)
    #   - num_iter   : RIDOTTO (~20 invece di ~30) perché ogni x-step è esatto
    #   - x_init     : AGGIUNTO — passa la FBP già calcolata (no doppio calcolo)
    print("-> Esecuzione PnP-HQS-CG...")
    t1 = time.time()
    pnp_recon = solver.reconstruct(
        sinogram=sinogram,
        projector=projector,
        x_init=fbp_recon,       # warm start dalla FBP
        num_iterations=num_iter,
        mu=mu,
        n_cg=n_cg
    )
    print(f"   [PnP-CG] Fatto in {time.time()-t1:.3f}s")

    return fbp_recon, pnp_recon


# ================================================================== #
#  MAIN
# ================================================================== #
if __name__ == "__main__":

    DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH_PESI            = "methods/pnp/pnp_denoiser_best_v2.pth"
    CARTELLA_SINOGRAMMI  = "data/sinogram_corrupted/test/angles_180"  # DA CAMBIARE
    CARTELLA_GROUND_TRUTH= "data/preprocessed/test"               # DA CAMBIARE
    CARTELLA_OUTPUT      = "data/pnp/results_pnp_180"                            # DA CAMBIARE

    print("--- INIZIALIZZAZIONE PIPELINE PnP-HQS-CG ---")
    print(f"    Device: {DEVICE}")

    solver = PnpHQS_Solver(denoiser_path=PATH_PESI, device=DEVICE)

    test_dataset = TCDataset(sino_dir=CARTELLA_SINOGRAMMI, gt_dir=CARTELLA_GROUND_TRUTH)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    os.makedirs(CARTELLA_OUTPUT, exist_ok=True)

    # Accumula metriche per la media finale
    risultati = []

    for i, (my_sinogram, clean_img) in enumerate(test_loader):

        my_sinogram = my_sinogram.to(DEVICE)
        clean_img   = clean_img.to(DEVICE)

        num_angoli = my_sinogram.shape[-2]
        my_angles  = np.linspace(0, np.pi, num_angoli, endpoint=False)

        # Seleziona parametri in base al numero di angoli
        params = PARAMS_PER_ANGOLO.get(
            num_angoli,
            {"num_iter": 15, "mu": 0.1, "n_cg": 5}  # fallback generico
        )

        print(f"\n{'='*45}")
        print(f"  Slice {i+1:3d} | angoli={num_angoli} | params={params}")
        print(f"{'='*45}")

        img_fbp, img_pnp = process_patient(
            sinogram=my_sinogram,
            angles=my_angles,
            solver=solver,
            **params
        )

        # --- Conversione numpy per metriche e salvataggio ---
        gt_np  = clean_img.cpu().numpy()[0, 0]
        fbp_np = img_fbp.cpu().numpy()[0, 0]
        pnp_np = img_pnp.cpu().numpy()[0, 0]

        # --- Metriche quantitative (richieste dalla consegna) ---
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

        # --- Salvataggio immagini (PnP per confronto visivo) ---
        np.save(os.path.join(CARTELLA_OUTPUT, f"pnp_{i+1:03d}.npy"), pnp_np)

    # --- Statistiche finali ---
    psnr_fbp_mean = np.mean([r["psnr_fbp"] for r in risultati])
    psnr_pnp_mean = np.mean([r["psnr_pnp"] for r in risultati])
    ssim_fbp_mean = np.mean([r["ssim_fbp"] for r in risultati])
    ssim_pnp_mean = np.mean([r["ssim_pnp"] for r in risultati])

    print(f"\n{'='*45}")
    print(f"  MEDIE FINALI su {len(risultati)} slice")
    print(f"{'='*45}")
    print(f"  [FBP]     PSNR={psnr_fbp_mean:.2f} dB  SSIM={ssim_fbp_mean:.4f}")
    print(f"  [PnP-CG]  PSNR={psnr_pnp_mean:.2f} dB  SSIM={ssim_pnp_mean:.4f}")

    # Salva CSV con tutte le metriche per slice
    import csv
    csv_path = os.path.join(CARTELLA_OUTPUT, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=risultati[0].keys())
        writer.writeheader()
        writer.writerows(risultati)
    print(f"\n  Metriche salvate in: {csv_path}")
    print(f"  Immagini salvate in: {CARTELLA_OUTPUT}/")
    print(f"\nFATTO! Elaborate {len(risultati)} slice.")

    
    
    

