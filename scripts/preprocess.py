#!/usr/bin/env python3
"""
Preprocessa il dataset Mayo per Sparse-view CT con split patient-level:
1. Train = tutti i pazienti in raw/train tranne validation_patient_id
2. Validation = un paziente scelto dentro raw/train
3. Test = tutto ciò che sta in raw/test
4. Resize 256x256 + normalizzazione [0,1]
5. Salvataggio in data/preprocessed/{split}/*.npy
6. Crea struttura directory per sinogrammi
7. Sanity check
"""

import os
import sys
import glob
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.preprocessing import get_transform
from data.dataset import MayoDatasetNpy

BASE_DATA_PATH = os.path.join(PROJECT_ROOT, "data")
DATASET_PATH = os.path.join(BASE_DATA_PATH, "raw")
PREPROCESSED_PATH = os.path.join(BASE_DATA_PATH, "preprocessed")
SINOGRAM_CLEAN_PATH = os.path.join(BASE_DATA_PATH, "sinogram_clean")
SINOGRAM_CORRUPTED_PATH = os.path.join(BASE_DATA_PATH, "sinogram_corrupted")

SPLITS = ["train", "validation", "test"]


def load_data_config():
    config_path = os.path.join(PROJECT_ROOT, "configs", "data_config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    required_keys = [
        "image_size",
        "angle_configs",
        "validation_patient_id",
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing key '{key}' in {config_path}")

    return config


def create_directory_structure(angle_configs):
    for split in SPLITS:
        os.makedirs(os.path.join(PREPROCESSED_PATH, split), exist_ok=True)

    for split in SPLITS:
        for n_angles in angle_configs:
            os.makedirs(
                os.path.join(SINOGRAM_CLEAN_PATH, split, f"angles_{n_angles}"),
                exist_ok=True
            )
            os.makedirs(
                os.path.join(SINOGRAM_CORRUPTED_PATH, split, f"angles_{n_angles}"),
                exist_ok=True
            )

    print("Directory structure created")


def get_patient_dirs(train_root):
    patient_dirs = [
        d for d in glob.glob(os.path.join(train_root, "*"))
        if os.path.isdir(d)
    ]
    patient_dirs = sorted(patient_dirs)

    if not patient_dirs:
        raise FileNotFoundError(f"No patient folders found in {train_root}")

    return patient_dirs


def collect_pngs_from_dir(directory):
    image_paths = sorted(
        glob.glob(os.path.join(directory, "**", "*.png"), recursive=True)
    )
    return image_paths


def split_by_patient(raw_train_path, raw_test_path, validation_patient_id):
    patient_dirs = get_patient_dirs(raw_train_path)

    train_patients = []
    val_patient = None

    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        if patient_id == validation_patient_id:
            val_patient = patient_dir
        else:
            train_patients.append(patient_dir)

    if val_patient is None:
        available = [os.path.basename(p) for p in patient_dirs]
        raise ValueError(
            f"Validation patient '{validation_patient_id}' not found in raw/train. "
            f"Available patients: {available}"
        )

    train_images = []
    for patient_dir in train_patients:
        train_images.extend(collect_pngs_from_dir(patient_dir))

    val_images = collect_pngs_from_dir(val_patient)
    test_images = collect_pngs_from_dir(raw_test_path)

    if not train_images:
        raise ValueError("No training images found after patient-level split.")
    if not val_images:
        raise ValueError("No validation images found for selected patient.")
    if not test_images:
        raise ValueError("No test images found in raw/test.")

    return train_images, val_images, test_images, train_patients, val_patient


def get_npy_filename(raw_path: str) -> str:
    rel_path = os.path.relpath(raw_path, DATASET_PATH)
    filename = rel_path.replace(os.sep, "_")
    return os.path.splitext(filename)[0] + ".npy"


def save_preprocessed_images(image_paths, split_name, transform):
    save_dir = os.path.join(PREPROCESSED_PATH, split_name)

    for img_path in image_paths:
        image = Image.open(img_path).convert("L")
        image_tensor = transform(image)
        image_np = image_tensor.squeeze().numpy().astype(np.float32)

        filename = get_npy_filename(img_path)
        save_path = os.path.join(save_dir, filename)
        np.save(save_path, image_np)

    print(f"Saved {split_name}: {len(image_paths)} images")


def sanity_check():
    train_npy = glob.glob(os.path.join(PREPROCESSED_PATH, "train", "*.npy"))
    val_npy = glob.glob(os.path.join(PREPROCESSED_PATH, "validation", "*.npy"))
    test_npy = glob.glob(os.path.join(PREPROCESSED_PATH, "test", "*.npy"))

    print("\nDATASET SUMMARY:")
    print(f"Train:      {len(train_npy)}")
    print(f"Validation: {len(val_npy)}")
    print(f"Test:       {len(test_npy)}")

    if train_npy:
        sample = np.load(train_npy[0])
        print(
            f"\nSample check: {sample.shape} | "
            f"[{sample.min():.3f}, {sample.max():.3f}] | {sample.dtype}"
        )

        train_loader = DataLoader(
            MayoDatasetNpy(train_npy),
            batch_size=8,
            shuffle=False
        )
        batch = next(iter(train_loader))
        print(f"Batch shape: {batch.shape} | dtype: {batch.dtype}")
    else:
        print("Warning: No training files found for DataLoader check")


def main():
    print("Starting Mayo dataset preprocessing with patient-level split...")

    config = load_data_config()

    image_size = config["image_size"]
    angle_configs = config["angle_configs"]
    validation_patient_id = config["validation_patient_id"]

    raw_train_path = os.path.join(DATASET_PATH, "train")
    raw_test_path = os.path.join(DATASET_PATH, "test")

    print(
        f"image_size={image_size}, "
        f"validation_patient_id={validation_patient_id}, "
        f"angle_configs={angle_configs}"
    )

    create_directory_structure(angle_configs)

    train_images, val_images, test_images, train_patients, val_patient = split_by_patient(
        raw_train_path=raw_train_path,
        raw_test_path=raw_test_path,
        validation_patient_id=validation_patient_id,
    )

    print("\nPATIENT SPLIT:")
    print(f"Validation patient: {os.path.basename(val_patient)}")
    print(f"Train patients: {[os.path.basename(p) for p in train_patients]}")
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Test images: {len(test_images)}")

    transform = get_transform(image_size)

    save_preprocessed_images(train_images, "train", transform)
    save_preprocessed_images(val_images, "validation", transform)
    save_preprocessed_images(test_images, "test", transform)

    sanity_check()

    print("\nPreprocessing COMPLETED!")


if __name__ == "__main__":
    main()