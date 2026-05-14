#!/usr/bin/env python3
"""
Preprocessa il dataset Mayo per Sparse-view CT:
1. Split train/val/test
2. Resize 256x256 + normalizzazione [0,1]
3. Salvataggio in data/preprocessed/{split}/*.npy
4. Crea struttura directory per sinogrammi
5. Sanity check
"""

import os
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from PIL import Image

from data.preprocessing import get_transform
from data.dataset import MayoDatasetNpy

BASE_DATA_PATH = "../data"
DATASET_PATH = os.path.join(BASE_DATA_PATH, "raw")
PREPROCESSED_PATH = os.path.join(BASE_DATA_PATH, "preprocessed")
SINOGRAM_CLEAN_PATH = os.path.join(BASE_DATA_PATH, "sinogram_clean")
SINOGRAM_CORRUPTED_PATH = os.path.join(BASE_DATA_PATH, "sinogram_corrupted")

splits = ["train", "validation", "test"]
angle_folders = ["angles_180", "angles_90", "angles_60", "angles_45"]


def create_directory_structure():
    """Crea tutta la struttura di directory necessaria"""
    # Preprocessed
    for split in splits:
        os.makedirs(os.path.join(PREPROCESSED_PATH, split), exist_ok=True)
    
    # Sinograms
    for split in splits:
        for angle_folder in angle_folders:
            os.makedirs(os.path.join(SINOGRAM_CLEAN_PATH, split, angle_folder), exist_ok=True)
            os.makedirs(os.path.join(SINOGRAM_CORRUPTED_PATH, split, angle_folder), exist_ok=True)
    
    print("✅ Directory structure created")


def get_npy_filename(raw_path: str) -> str:
    """Converte path raw in nome file .npy univoco"""
    rel_path = os.path.relpath(raw_path, DATASET_PATH)
    filename = rel_path.replace(os.sep, "_")
    return os.path.splitext(filename)[0] + ".npy"


def save_preprocessed_images(image_paths: list, split_name: str, transform):
    """Salva immagini preprocessate come .npy"""
    save_dir = os.path.join(PREPROCESSED_PATH, split_name)
    
    for img_path in image_paths:
        image = Image.open(img_path).convert("L")
        image_tensor = transform(image)
        image_np = image_tensor.squeeze().numpy()
        
        filename = get_npy_filename(img_path)
        save_path = os.path.join(save_dir, filename)
        np.save(save_path, image_np)
    
    print(f"✅ Saved {split_name}: {len(image_paths)} images")


def sanity_check():
    """Verifica che il preprocessing sia corretto"""
    train_npy = glob.glob(os.path.join(PREPROCESSED_PATH, "train", "*.npy"))
    val_npy = glob.glob(os.path.join(PREPROCESSED_PATH, "validation", "*.npy"))
    test_npy = glob.glob(os.path.join(PREPROCESSED_PATH, "test", "*.npy"))
    
    # Conta file
    print("\n📊 DATASET SUMMARY:")
    print(f"Train:    {len(train_npy)}")
    print(f"Val:      {len(val_npy)}")
    print(f"Test:     {len(test_npy)}")
    
    # Check un campione
    if train_npy:
        sample = np.load(train_npy[0])
        print(f"\n✅ Sample check: {sample.shape} | [{sample.min():.3f}, {sample.max():.3f}] | {sample.dtype}")
    
    # Test DataLoader
    train_loader = DataLoader(MayoDatasetNpy(train_npy), batch_size=8, shuffle=False)
    batch = next(iter(train_loader))
    print(f"✅ Batch shape: {batch.shape} | dtype: {batch.dtype}")


def main():
    """Pipeline principale di preprocessing"""
    print("🚀 Starting Mayo dataset preprocessing...")
    
    # 1. Crea directory
    create_directory_structure()
    
    # 2. Trova immagini raw
    all_images = glob.glob(os.path.join(DATASET_PATH, "**", "*.png"), recursive=True)
    if not all_images:
        raise FileNotFoundError(f"No PNG files found in {DATASET_PATH}")
    
    print(f"📁 Found {len(all_images)} raw images")
    
    # 3. Split dataset (70/15/15)
    train_raw, temp_raw = train_test_split(
        all_images, test_size=0.3, random_state=42
    )
    val_raw, test_raw = train_test_split(
        temp_raw, test_size=0.5, random_state=42
    )
    
    # 4. Preprocessa e salva
    transform = get_transform(256)
    save_preprocessed_images(train_raw, "train", transform)
    save_preprocessed_images(val_raw, "validation", transform)
    save_preprocessed_images(test_raw, "test", transform)
    
    # 5. Sanity check
    sanity_check()
    
    print("\n🎉 Preprocessing COMPLETED!")


if __name__ == "__main__":
    main()