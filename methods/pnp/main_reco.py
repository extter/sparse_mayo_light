import torch
import numpy as np 
import matplotlib.pyplot as plt 

from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

from methods.pnp.dataset import TCDataset
from methods.pnp.hqs import PnpHQS_Solver
from third_party.ippy.operators import CTProjector

def process_patient(sinogram, angles, solver, num_iter=15, mu=0.05, step=0.02):
    """
    Funzione slim che prende QUALSIASI sinogramma e angoli,
    crea l'operatore fisico al volo e restituisce le ricostruzioni FBP e PnP.
    """
    device = sinogram.device
    
    # 1. Image shape dectected from sinogram shape
    det_count = sinogram.shape[-1]
    img_shape = (det_count, det_count)
    
    # 2. CT projcetor inizialization
    projector = CTProjector(img_shape=img_shape, angles=angles, geometry="parallel", force_cpu=False)
    
    # 3. Baseline FBP
    print("-> Esecuzione FBP...")
    with torch.no_grad():
        fbp_recon = projector.FBP(sinogram)
        fbp_recon = torch.clamp(fbp_recon, 0.0, 1.0)
        
    # 4. PNP recostruction
    print("-> Esecuzione PnP-HQS...")
    pnp_recon = solver.reconstruct(
        sinogram=sinogram, 
        projector=projector, 
        num_iterations=num_iter, 
        mu=mu, 
        step=step
    )
    
    return fbp_recon, pnp_recon

# ==========================================
# ESECUZIONE PRINCIPALE (MAIN)
# ==========================================
if __name__ == "__main__":
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH_PESI = "pnp_denoiser_best.pth"
    CARTELLA_SINOGRAMMI = "/kaggle/input/mayo-dataset/test/sinograms_noisy"
    CARTELLA_GROUND_TRUTH = "/kaggle/input/mayo-dataset/test/images_clean"

    print("--- INIZIALIZZAZIONE PIPELINE ---")
    
    # 1. Carichiamo il TUO Solver (viene fatto una volta sola per tutti i pazienti)
    solver = PnpHQS_Solver(denoiser_path=PATH_PESI, device=DEVICE)

    # 2. Carichiamo i dati di Test
    test_dataset = TCDataset(sino_dir=CARTELLA_SINOGRAMMI, gt_dir=CARTELLA_GROUND_TRUTH)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for i, (my_sinogram, clean_img) in enumerate(test_loader):
        
        # Spostiamo i tensori sulla GPU
        my_sinogram = my_sinogram.to(DEVICE)
        clean_img = clean_img.to(DEVICE)
        
        # Calcoliamo gli angoli dinamicamente basandoci sulla forma del sinogramma caricato
        num_angoli = my_sinogram.shape[-2]
        my_angles = np.linspace(0, np.pi, num_angoli, endpoint=False)
        
        print(f"\n--- Elaborazione Slice {i+1} ---")

        # 4. LA CHIAMATA SLIM 
        img_fbp, img_pnp = process_patient(
            sinogram=my_sinogram, 
            angles=my_angles, 
            solver=solver, 
            num_iter=15
        )

        # 5. METRICHE E PLOT
        # Estrazione array 2D per i calcoli
        gt_np = clean_img.cpu().numpy()[0, 0]
        fbp_np = img_fbp.cpu().numpy()[0, 0]
        pnp_np = img_pnp.cpu().numpy()[0, 0]
        sino_np = my_sinogram.cpu().numpy()[0, 0]

        psnr_fbp = 20 * np.log10(1.0 / np.sqrt(np.mean((fbp_np - gt_np) ** 2)))
        psnr_pnp = 20 * np.log10(1.0 / np.sqrt(np.mean((pnp_np - gt_np) ** 2)))
        ssim_fbp = ssim(gt_np, fbp_np, data_range=1.0)
        ssim_pnp = ssim(gt_np, pnp_np, data_range=1.0)

        print(f"Risultati Slice {i+1}:")
        print(f"FBP -> PSNR: {psnr_fbp:.2f} dB | SSIM: {ssim_fbp:.4f}")
        print(f"PnP -> PSNR: {psnr_pnp:.2f} dB | SSIM: {ssim_pnp:.4f}")

        # Plot
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(gt_np, cmap='gray', vmin=0, vmax=1); axes[0].set_title("Ground Truth")
        axes[1].imshow(sino_np, cmap='gray'); axes[1].set_title("Sinogramma Rumoroso")
        axes[2].imshow(fbp_np, cmap='gray', vmin=0, vmax=1); axes[2].set_title(f"FBP (PSNR: {psnr_fbp:.1f})")
        axes[3].imshow(pnp_np, cmap='gray', vmin=0, vmax=1); axes[3].set_title(f"PnP-HQS (PSNR: {psnr_pnp:.1f})")
        for ax in axes: ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"final_results_slice_{i+1}.png", dpi=300)
        plt.show()
        
        # Mettiamo un BREAK qui così elabora solo la prima immagine per farti vedere se tutto funziona.
        # Quando vorrai elaborare l'intero dataset, ti basterà cancellare o commentare questa riga.
        break
    
    
    

