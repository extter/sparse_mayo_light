# dataset.py - versione migliorata
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import os

class MayoDataset(Dataset):
    """Carica immagini raw (.png) e applica transform"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        
        image = Image.open(path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image

class MayoDatasetNpy(Dataset):
    """Carica immagini già preprocessate (.npy), float32 [0,1]"""
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Numpy file not found: {path}")
        
        image_np = np.load(path)  # [256, 256]
        # Esplicito controllo dtype e range
        assert image_np.dtype == np.float32, f"Expected float32, got {image_np.dtype}"
        assert (image_np >= 0).all() and (image_np <= 1).all(), "Values must be in [0,1]"
        
        return torch.from_numpy(image_np).unsqueeze(0).float()  # [1, 256, 256]