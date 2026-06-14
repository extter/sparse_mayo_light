import os 
import torch 
import torch.nn as nn 
import numpy as np 

from methods.end_to_end_2.unet import SimpleUNet

class PnpHQS_Solver(nn.Module):
    def __init__(self, denoiser_path, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.denoiser = SimpleUNet(in_ch=1, out_ch=1, base_ch=32).to(self.device)
        self.denoiser.load_state_dict(torch.load(denoiser_path, map_location=self.device))
        self.denoiser.eval()
        
        for param in self.denoiser.parameters():
            param.requires_grad = False

    def reconstruct(self, sinogram, projector, num_iterations=30, mu=0.5, step_size=0.1):
        """
        - step_size: Il learning rate per la discesa del gradiente.
        - mu: Quanto pesa l'accoppiamento con il denoiser (z) rispetto alla fisica.
        """
        with torch.no_grad():
            y = sinogram.to(self.device)
            
            # Partiamo da una FBP (se disponibile) per non partire da zero
            try:
                x = projector.FBP(y).to(self.device)
            except AttributeError:
                x = projector.T(y).to(self.device)
                
            x = torch.clamp(x, 0.0, 1.0)
            z = x.clone()

            print(f'   [HQS Base] Avvio {num_iterations} iterazioni (mu={mu}, step={step_size})...')
            ones = torch.ones_like(x)
            ATA_norm = projector.T(projector(ones)).max().item() + 1e-8
            
            for k in range(num_iterations):
                # FASE 1: Discesa del Gradiente
                Ax = projector(x)
                errore_fisico = Ax - y
                
                # Dividere per ATA_NORM impedisce esplosione
                grad_physics = projector.T(errore_fisico) / ATA_norm
                
                grad_reg = mu * (x - z)
                
                gradiente_totale = grad_physics + grad_reg
                x = x - step_size * gradiente_totale
                
                x = torch.clamp(x, 0.0, 1.0)

                # --- FASE 2: Denoising (Plug and Play) ---
                with torch.amp.autocast('cuda'):
                    z = self.denoiser(x)
                    
                z = torch.clamp(z, 0.0, 1.0)
                
            return z
                



