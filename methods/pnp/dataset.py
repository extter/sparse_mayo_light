import os 
import json
import numpy as np 
import torch 
from torch.utils.data import Dataset

class GaussianNoiseDataset(Dataset):
    def __init__ (self, gt_dir, split="train"):
        self.gtdir = gt_dir
        self.split = split
        
        self.filenames = sorted([f for f in os.listdir(self.gtdir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        file_path = os.path.join(self.gtdir, filename)

        clean_image = np.load(file_path)
        clean_tensor = torch.from_numpy(clean_image).float().unsqueeze(0) # Shape: (1,256,256)

        if self.split == 'train':
            sigma = np.random.uniform(0.01, 0.05)
            noise = torch.randn_like(clean_tensor) * sigma
            noisy_tensor = clean_tensor + noise
        else:
            noise = torch.randn_like(clean_tensor) * 0.03
            noisy_tensor = clean_tensor + noise

        return noisy_tensor, clean_tensor
    
    
        
            


    