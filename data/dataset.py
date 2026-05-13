# Dataset class and DataLoader setup
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np

class MayoDataset(Dataset):
    """Carica immagini raw (.png) e applica transform"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
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
        image_np = np.load(self.image_paths[idx])          # [256, 256]
        return torch.from_numpy(image_np).unsqueeze(0)     # [1, 256, 256]