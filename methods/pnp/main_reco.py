import torch
import time
import numpy as np 
import matplotlib.pyplot as plt 

from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from methods.pnp.dataset import TCDataset
from methods.pnp.hqs import PnpHQS_Solver
from third_party.ippy.operators import CTProjector

def process_patient(sinogram, angles, solver, num_iter=100, mu=0.5, step=1):
    device = sinogram.device
    
    det_count = 512
    img_shape = (256,256)
    
    projector = CTProjector(
        img_shape=img_shape, 
        angles=angles, 
        det_size=det_count,
        geometry="parallel", 
        force_cpu=False
    )
    
    # Baseline FBP
    print("-> Esecuzione FBP...")
    t0 = time.time()
    with torch.no_grad():
        fbp_recon = projector.FBP(sinogram)
        fbp_recon = torch.clamp(fbp_recon, 0.0, 1.0)
    print(f"   [FBP] Fatto in {time.time()-t0:.3f}s")
        
    # PNP recostruction
    print(f"-> Esecuzione PnP-HQS...")
    t1 = time.time()
    pnp_recon = solver.reconstruct(
        sinogram=sinogram, 
        projector=projector, 
        num_iterations=num_iter, 
        mu=mu, 
        step_size=step
    )
    print(f"   [PnP] Fatto in {time.time()-t1:.3f}s")
    
    return fbp_recon, pnp_recon

# ==========================================
# ESECUZIONE PRINCIPALE (MAIN)
# ==========================================
if __name__ == "__main__":
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH_PESI = "pnp_denoiser_best.pth" # DA CAMBIARE 
    CARTELLA_SINOGRAMMI = "data/sinogram_corrupted/test/angles_45" # DA CAMBIARE ANCHE A SECONDA ANGOLO
    CARTELLA_GROUND_TRUTH = "data/preprocessed/test" # DA CAMBIARE

    print("--- INIZIALIZZAZIONE PIPELINE BASE ---")
    
    solver = PnpHQS_Solver(denoiser_path=PATH_PESI, device=DEVICE)

    test_dataset = TCDataset(sino_dir=CARTELLA_SINOGRAMMI, gt_dir=CARTELLA_GROUND_TRUTH)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    CARTELLA_OUTPUT = "./data/pnp/reco_pnp_45" # DA CAMBIARE 
    os.makedirs(CARTELLA_OUTPUT, exist_ok=True)

    for i, (my_sinogram, clean_img) in enumerate(test_loader):
        
        my_sinogram = my_sinogram.to(DEVICE)
        clean_img = clean_img.to(DEVICE)
        
        num_angoli = my_sinogram.shape[-2]
        my_angles = np.linspace(0, np.pi, num_angoli, endpoint=False)
        
        print(f"\n{'='*40}")
        print(f"--- Elaborazione Slice {i+1} ---")
        print(f"{'='*40}")

        # Configurazione Base: a seconda dell'angolo necessito differente mu e step
        img_fbp, img_pnp = process_patient(
            sinogram=my_sinogram, 
            angles=my_angles, 
            solver=solver, 
            num_iter=30, 
            mu=0.1, 
            step=1.5 
        )
        
        pnp_np = img_pnp.cpu().numpy()[0, 0]

        # Salvataggio diretto col numero sequenziale
        nome_salvataggio = os.path.join(CARTELLA_OUTPUT, f"pnp_{i+1:03d}.npy")
        np.save(nome_salvataggio, pnp_np)

print(f"FATTO! Salvate {len(test_loader)} immagini PnP.")

    
    
    

