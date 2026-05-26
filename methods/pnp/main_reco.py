import numpy as np 
import torch 
import matplotlib.pyplot as pyplot

from third_party.ippy.operators import *
from methods.pnp.hqs import PnpHQS_Solver

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # GEOMETRIC CONFIGURATION 
    IMG_SHAPE = (256, 256)
    DET_SIZE = 256
    N_ANGLES = 180

    angles = np.linspace(0, np.pi, N_ANGLES, endpoint=False)

    projector = CTProjector(
        img_shape=IMG_SHAPE,
          det_size=DET_SIZE, 
          angles=angles, 
          force_cpu=False
          )
    
    
    

