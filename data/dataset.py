# dataset.py - versione migliorata
from torch.utils.data import Dataset,  DataLoader
from PIL import Image
import torch
import numpy as np
from glob import glob
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
    


class SinogramDataset(Dataset):
    """
    Carica triple:
    - sinogramma corrotto
    - sinogramma clean
    - immagine ground truth preprocessata

    Returns:
        corrupted, clean, x_true, fname
    con shape:
        corrupted: [1, det, angles]
        clean:     [1, det, angles]
        x_true:    [1, H, W]
    """
    def __init__(self, clean_paths, corrupted_paths, image_paths):
        self.clean_paths = sorted(clean_paths)
        self.corrupted_paths = sorted(corrupted_paths)
        self.image_paths = sorted(image_paths)

        assert len(self.clean_paths) == len(self.corrupted_paths), (
            "Mismatch between clean and corrupted sinograms."
        )
        assert len(self.clean_paths) == len(self.image_paths), (
            "Mismatch between sinograms and ground truth images."
        )

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_path = self.clean_paths[idx]
        corrupted_path = self.corrupted_paths[idx]
        image_path = self.image_paths[idx]

        for path in [clean_path, corrupted_path, image_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

        clean = np.load(clean_path).astype(np.float32)
        corrupted = np.load(corrupted_path).astype(np.float32)
        x_true = np.load(image_path).astype(np.float32)

        clean = torch.from_numpy(clean).unsqueeze(0)
        corrupted = torch.from_numpy(corrupted).unsqueeze(0)
        x_true = torch.from_numpy(x_true).unsqueeze(0)

        fname = os.path.basename(image_path)
        return corrupted, clean, x_true, fname


def get_sinogram_paths(base_data_path, split, n_angles):
    """
    Restituisce le liste di path per immagini preprocessate,
    sinogrammi clean e sinogrammi corrupted.

    Args:
        base_data_path (str): path root di data/
        split (str): train / validation / test
        n_angles (int): numero di angoli

    Returns:
        tuple[list[str], list[str], list[str]]
    """
    image_paths = sorted(
        glob(os.path.join(base_data_path, "preprocessed", split, "*.npy"))
    )

    clean_paths = sorted(
        glob(
            os.path.join(
                base_data_path,
                "sinogram_clean",
                split,
                f"angles_{n_angles}",
                "*.npy",
            )
        )
    )

    corrupted_paths = sorted(
        glob(
            os.path.join(
                base_data_path,
                "sinogram_corrupted",
                split,
                f"angles_{n_angles}",
                "*.npy",
            )
        )
    )

    return image_paths, clean_paths, corrupted_paths


def create_sinogram_dataloaders(base_data_path, angle_configs, batch_size=1):
    """
    Crea dataset e dataloader per tutti gli split e tutte le configurazioni angolari.

    Args:
        base_data_path (str): path root di data/
        angle_configs (list[int]): es. [180, 90, 60, 45]
        batch_size (int): batch size dei DataLoader

    Returns:
        tuple[dict, dict]:
            datasets[split][n_angles]
            dataloaders[split][n_angles]
    """
    datasets = {
        "train": {},
        "validation": {},
        "test": {},
    }

    dataloaders = {
        "train": {},
        "validation": {},
        "test": {},
    }

    for split in ["train", "validation", "test"]:
        for n_angles in angle_configs:
            image_paths, clean_paths, corrupted_paths = get_sinogram_paths(
                base_data_path=base_data_path,
                split=split,
                n_angles=n_angles,
            )

            dataset = SinogramDataset(
                clean_paths=clean_paths,
                corrupted_paths=corrupted_paths,
                image_paths=image_paths,
            )

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
            )

            datasets[split][n_angles] = dataset
            dataloaders[split][n_angles] = loader

    return datasets, dataloaders