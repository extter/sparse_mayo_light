import os 
import torch 
import torch.nn as nn 
import numpy as np 

from methods.end_to_end.unet import UNet


class PnpHQS_Solver(nn.Module):
    def __init__(self, denoiser_path, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.denoiser = UNet(ch_in=1, ch_out=1, middle_ch=(32, 64, 128, 256)).to(self.device)
        self.denoiser.load_state_dict(torch.load(denoiser_path, map_location=device))
        self.denoiser().to(device)
        
        self.denoiser.eval()

        for param in self.denoiser.parameters():
            param.requires_grad = False

    def reconstruct(self, sinogram, projector, num_iterations=20, mu=0.1, step=0.01):
        with torch.no_grad():
            y = sinogram.to(self.device)
            
            # Initialiazition (Starting from noisy_sinogram)
            x = projector.FBP(y).to(self.device)

            z = x.clone()
            
            print('Starting HQS cycle')

            for k in range(num_iterations):
                
                # STEP 1 : DATA FIDELITY
                Ax = projector(x)

                sino_error = Ax - y

                grad_physics = projector.T(sino_error)
                grad_coupling = mu * (x - z)
                grad = grad_physics + grad_coupling

                x = x - step * grad

                x = torch.clamp(x, min=0.0, max=1.0)

                # STEP 2: PNP DENOISING (Trained UNet)
                with torch.no_grad():
                    z = self.denoiser(x)

            return z

                



